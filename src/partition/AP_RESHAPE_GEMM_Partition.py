from . import *
from onnx import helper


def AP_RESHAPE_GEMM_Partition (
        partition:int, model:onnx.ModelProto, 
        apl_node_name:str, 
        reshape_node_name:str, 
        gemm_node_name:str, 
        splt_node_name:str, 
        sums_node_name:str) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    apl_index = 0
    reshape_index = 0
    gemm_index = 0
    apl_node = None
    reshape_node = None
    gemm_node = None

    for i, node in enumerate(model.graph.node):
        if node.name == apl_node_name:
            apl_node, apl_index = node, i
        if node.name == reshape_node_name:
            reshape_node, reshape_index = node, i
        if node.name == gemm_node_name:
            gemm_node, gemm_index = node, i
            
    aplNameList = []
    gemmNameList = []
    # GAP - Input
    tensor_idx = 0
    apl_input = None
    apl_input_Name = []
    tensor_idx, apl_input = util.get_value_info(tensor_name=apl_node.input[0],model=model)
    apl_input_dims = [dims.dim_value for dims in apl_input.type.tensor_type.shape.dim]
    number = util.output_shape(partition_num=partition,shape=apl_input_dims[1])

    for i in range(partition):   
        name = apl_input.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=apl_input)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        apl_input_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(apl_input)


    # GlobalAveragePool - Output
    tensor_idx = 0
    apl_output = None
    apl_output_Name = []
    tensor_idx, apl_output = util.get_value_info(tensor_name=apl_node.output[0],model=model)
    for i in range(partition):  
        name = apl_output.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=apl_output)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        apl_output_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(apl_output)  

    # Flatten - Output
    tensor_idx = 0
    reshape_output = None
    reshape_output_Name = []
    tensor_idx, reshape_output = util.get_value_info(tensor_name=reshape_node.output[0],model=model)

    for i in range(partition):  
        name = reshape_output.name + f"/split_{i}"
        elem_type = reshape_output.type.tensor_type.elem_type
        shape = [dims.dim_value for dims in reshape_output.type.tensor_type.shape.dim]
        shape[1] = number[i]
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        reshape_output_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(reshape_output)

    # Gemm - Matrix B (weight)
    tensor_idx = 0
    gemm_weight = None
    for tensor_info, initializer in enumerate(model.graph.initializer):
        if initializer.name == gemm_node.input[1]:
            gemm_weight = initializer
            tensor_idx = tensor_info
            break
    weight = numpy_helper.to_array(gemm_weight)
    gemm_weight_Name = []
    for i in range(partition):
        dims = list(weight.shape)
        minn = sum(number[0:i])
        maxx = sum(number[0:i+1])
        dims[1] = maxx - minn
        vals = weight[:,minn:maxx]
        name = gemm_weight.name + f"/split_{i}"
        tensor = onnx.helper.make_tensor(
                name=name,
                data_type=gemm_weight.data_type,
                dims=dims, vals=vals)
        gemm_weight_Name.append(name)
        model.graph.initializer.insert(tensor_idx + 1 + i,tensor)
    model.graph.initializer.remove(gemm_weight)      
    # Gemm - Matrix C (bias)
    tensor_idx = 0
    gemm_bias = None
    if len(gemm_node.input) == 3:
        for tensor_info, initializer in enumerate(model.graph.initializer):
            if initializer.name == gemm_node.input[2]:
                gemm_bias = initializer
                tensor_idx = tensor_info
                break
    # Gemm - Matrix Y (Output)
    tensor_idx = 0
    gemm_output = None
    gemm_output_Name = []

    gemm_output, tensor_idx
    tensor_idx, gemm_output = util.get_value_info(tensor_name=gemm_node.output[0],model=model)
    if tensor_idx == -1: tensor_idx = len(model.graph.value_info)

    for i in range(partition):  
        name = gemm_output.name + f"/split_{i}"
        elem_type = gemm_output.type.tensor_type.elem_type
        shape = [dims.dim_value for dims in gemm_output.type.tensor_type.shape.dim]
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        gemm_output_Name.append(name)
        if tensor_idx == -1:
            tensor_idx = len(model.graph.value_info)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)

    ##### Node
    for i, node in enumerate(model.graph.node):
        if node.name == apl_node_name:
            apl_node, apl_index = node, i

    # Create Split Node
    split = onnx.helper.make_node(
        'Split',
        name=splt_node_name,
        inputs=apl_node.input,
        outputs=apl_input_Name,
        axis=1,
    )   
    model.graph.node.insert(apl_index, split)
    
    apl_attrs = {attr.name: helper.get_attribute_value(attr) for attr in apl_node.attribute}

    # Create AveragePool Node
    for i, node in enumerate(model.graph.node):
        if node.name == apl_node_name:
            apl_node, apl_index = node, i
    for i in range(partition):       
        apl = onnx.helper.make_node(
                        'AveragePool',
                        name=apl_node.name + f"_{i}", 
                        inputs=[apl_input_Name[i]],
                        outputs=[apl_output_Name[i]],
                        **apl_attrs
                        )   
        model.graph.node.insert(apl_index + 1 + i, apl)
        aplNameList.append(apl.name)
    model.graph.node.remove(apl_node)

    # Create Flatten Node
    for i, node in enumerate(model.graph.node):
        if node.name == reshape_node_name:
            reshape_node, reshape_index = node, i
    for i in range(partition): 
        reshape = onnx.helper.make_node(
                        'Flatten',
                        name=reshape_node.name + f"_{i}", 
                        inputs=[apl_output_Name[i]],
                        outputs=[reshape_output_Name[i]],
                        )
        model.graph.node.insert(reshape_index + 1 + i, reshape) 
    model.graph.node.remove(reshape_node)

    # Create Gemm Node
    for i, node in enumerate(model.graph.node):
        if node.name == gemm_node_name:
            gemm_node, gemm_index = node, i
            
    def get_gemm_attributes(attrs: dict, has_bias: bool) -> dict:
        """根據是否有 bias 決定要不要包含 beta，回傳 Gemm node 可接受的屬性 dict"""
        return {
            key: value for key, value in attrs.items()
            if key in {"alpha", "transB"} or (key == "beta" and has_bias)
        }

    # 預先擷取 Gemm 屬性為字典
    gemm_attrs_raw = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in gemm_node.attribute
    }

    for i in range(partition):
        has_bias = (i == 0 and gemm_bias is not None)

        # 組成輸入
        inputs = [reshape_output_Name[i], gemm_weight_Name[i]]
        if has_bias:
            inputs.append(gemm_bias.name)

        # 根據是否有 bias 決定選用哪些屬性
        gemm_attributes = get_gemm_attributes(gemm_attrs_raw, has_bias)

        # 建立新的 Gemm 節點
        gemm = onnx.helper.make_node(
            'Gemm',
            name=f"{gemm_node.name}_{i}",
            inputs=inputs,
            outputs=[gemm_output_Name[i]],
            **gemm_attributes
        )

        model.graph.node.insert(gemm_index + 1 + i, gemm)
        gemmNameList.append(gemm.name)

    model.graph.node.remove(gemm_node)
    add = onnx.helper.make_node(
                        'Sum',
                        name=sums_node_name,
                        inputs=gemm_output_Name,
                        outputs=[gemm_node.output[0]]
                        ) 
    model.graph.node.insert(gemm_index + partition , add)

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    return model, aplNameList, gemmNameList
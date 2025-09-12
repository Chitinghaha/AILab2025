from . import *

def GAP_FLATTEN_GEMM_Patrition (
        partition:int, model:onnx.ModelProto, 
        gapl_node_name:str, 
        fltn_node_name:str, 
        gemm_node_name:str, 
        splt_node_name:str, 
        sums_node_name:str) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    gapl_index = 0
    fltn_index = 0
    gemm_index = 0
    gapl_node = None
    fltn_node = None
    gemm_node = None

    for i, node in enumerate(model.graph.node):
        if node.name == gapl_node_name:
            gapl_node, gapl_index = node, i
        if node.name == fltn_node_name:
            fltn_node, fltn_index = node, i
        if node.name == gemm_node_name:
            gemm_node, gemm_index = node, i
            
    gaplNameList = []
    gemmNameList = []
    # GAP - Input
    tensor_idx = 0
    gapl_input = None
    gapl_input_Name = []
    tensor_idx, gapl_input = util.get_value_info(tensor_name=gapl_node.input[0],model=model)
    gapl_input_dims = [dims.dim_value for dims in gapl_input.type.tensor_type.shape.dim]
    number = util.output_shape(partition_num=partition,shape=gapl_input_dims[1])

    for i in range(partition):   
        name = gapl_input.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=gapl_input)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        gapl_input_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(gapl_input)


    # GlobalAveragePool - Output
    tensor_idx = 0
    gapl_output = None
    gapl_output_Name = []
    tensor_idx, gapl_output = util.get_value_info(tensor_name=gapl_node.output[0],model=model)
    for i in range(partition):  
        name = gapl_output.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=gapl_output)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        gapl_output_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(gapl_output)  

    # Flatten - Output
    tensor_idx = 0
    fltn_output = None
    fltn_output_Name = []
    tensor_idx, fltn_output = util.get_value_info(tensor_name=fltn_node.output[0],model=model)

    for i in range(partition):  
        name = fltn_output.name + f"/split_{i}"
        elem_type = fltn_output.type.tensor_type.elem_type
        shape = [dims.dim_value for dims in fltn_output.type.tensor_type.shape.dim]
        shape[1] = number[i]
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(name=name,type_proto=tensor_type_proto)
        fltn_output_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    if tensor_idx != -1:
        model.graph.value_info.remove(fltn_output)

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
        if node.name == gapl_node_name:
            gapl_node, gapl_index = node, i

    # Create Split Node
    split = onnx.helper.make_node(
        'Split',
        name=splt_node_name,
        inputs=gapl_node.input,
        outputs=gapl_input_Name,
        axis=1,
    )   
    model.graph.node.insert(gapl_index, split)

    # Create GlobalAveragePool Node
    for i, node in enumerate(model.graph.node):
        if node.name == gapl_node_name:
            gapl_node, gapl_index = node, i
    for i in range(partition):       
        gapl = onnx.helper.make_node(
                        'GlobalAveragePool',
                        name=gapl_node.name + f"_{i}", 
                        inputs=[gapl_input_Name[i]],
                        outputs=[gapl_output_Name[i]],
                        )   
        model.graph.node.insert(gapl_index + 1 + i, gapl)
        gaplNameList.append(gapl.name)
    model.graph.node.remove(gapl_node)

    # Create Flatten Node
    for i, node in enumerate(model.graph.node):
        if node.name == fltn_node_name:
            fltn_node, fltn_index = node, i
    for i in range(partition): 
        fltn = onnx.helper.make_node(
                        'Flatten',
                        name=fltn_node.name + f"_{i}", 
                        inputs=[gapl_output_Name[i]],
                        outputs=[fltn_output_Name[i]],
                        )
        model.graph.node.insert(fltn_index + 1 + i, fltn) 
    model.graph.node.remove(fltn_node)

    # Create Gemm Node
    for i, node in enumerate(model.graph.node):
        if node.name == gemm_node_name:
            gemm_node, gemm_index = node, i
    for i in range(partition):
        if i == 0:
            gemm = onnx.helper.make_node(
                        'Gemm',
                        name=gemm_node.name + f"_{i}", 
                        inputs=[fltn_output_Name[i], gemm_weight_Name[i], gemm_bias.name],
                        outputs=[gemm_output_Name[i]],
                        alpha=gemm_node.attribute[0].f,
                        beta=gemm_node.attribute[1].f,
                        transB=gemm_node.attribute[2].i,
                        )
        else:
            gemm = onnx.helper.make_node(
                        'Gemm',
                        name=gemm_node.name + f"_{i}", 
                        inputs=[fltn_output_Name[i], gemm_weight_Name[i]],
                        outputs=[gemm_output_Name[i]],
                        alpha=gemm_node.attribute[0].f,
                        transB=gemm_node.attribute[2].i,
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

    return model, gaplNameList, gemmNameList
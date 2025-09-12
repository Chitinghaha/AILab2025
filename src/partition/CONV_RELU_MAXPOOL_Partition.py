from . import *
from onnx import helper

def CONV_RELU_MAXPOOL_Partition(partition:int, model:onnx.ModelProto, conv_node_name:str, relu_node_name:str, mxpo_node_name:str,concat_node_name:str) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    conv_index = 0
    relu_index = 0
    mxpo_index = 0
    conv_node = None
    relu_node = None
    mxpo_node = None
    convNameList = []
    maxPoolNameList = []    


    for i, node in enumerate(model.graph.node):
        if node.name == conv_node_name:
            conv_node = node
            conv_index = i
        if node.name == relu_node_name:
            relu_node = node
            relu_index = i
        if node.name == mxpo_node_name:
            mxpo_node = node
            mxpo_index = i

    # Weight
    tensor_idx = 0
    conv_weight = None
    for tensor_info, initializer in enumerate(model.graph.initializer):
        if initializer.name == conv_node.input[1]:
            conv_weight = initializer
            tensor_idx = tensor_info
            break
    kernel = numpy_helper.to_array(conv_weight)
    kernel_Name = []
    number = util.output_shape(partition_num=partition,shape=kernel.shape[0])
    for i in range(partition):
        dims, vals = util.partition_numpy_kernel(number=number, partition_idx=i, tensor=kernel)
        name = conv_weight.name + f"/split_{i}"
        tensor = onnx.helper.make_tensor(
            name=name,
            data_type=conv_weight.data_type,
            dims=dims, vals=vals)
        kernel_Name.append(name)
        model.graph.initializer.insert(tensor_idx + 1 + i,tensor)
    model.graph.initializer.remove(conv_weight)

    # Bias
    tensor_idx = 0
    conv_bias = None
    if len(conv_node.input) == 3:
        for tensor_info, initializer in enumerate(model.graph.initializer):
            if initializer.name == conv_node.input[2]:
                conv_bias = initializer
                tensor_idx = tensor_info
                break
        bias = None if conv_bias == None else numpy_helper.to_array(conv_bias)
        bias_Name = []
        for i in range(partition):
            name = conv_bias.name + f"/split_{i}"
            shape = bias.shape
            minn = sum(number[0:i])
            maxx = sum(number[0:i+1])
            tensor = onnx.helper.make_tensor(
                name=name,
                data_type=conv_bias.data_type,
                dims=[maxx-minn], vals=bias[minn:maxx])
            bias_Name.append(name)
            model.graph.initializer.insert(tensor_idx + 1 + i, tensor)
        model.graph.initializer.remove(conv_bias)

    # Create Value_info
    # CONV - Output
    tensor_idx = 0
    conv_Name = []
    conv_output = None
    tensor_idx, conv_output = util.get_value_info(tensor_name=conv_node.output[0],model=model)
    
    for i in range(partition):
        name = conv_output.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=conv_output)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(
            name=name,
            type_proto=tensor_type_proto
            )
        conv_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    model.graph.value_info.remove(conv_output)

    # RELU - Output
    tensor_idx = 0
    relu_Name = []
    relu_output = None
    tensor_idx, relu_output = util.get_value_info(tensor_name=relu_node.output[0],model=model)
    
    for i in range(partition):   
        name = relu_output.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=relu_output)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(
            name=name,
            type_proto=tensor_type_proto
            )
        relu_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    model.graph.value_info.remove(relu_output)


    # MaxPool - Output
    tensor_idx = 0
    mxpo_Name = []
    mxpo_output = None
    tensor_idx, mxpo_output = util.get_value_info(tensor_name=mxpo_node.output[0],model=model)
    
    for i in range(partition):   
        name = mxpo_output.name + f"/split_{i}"
        elem_type, shape = util.partition_output(number=number, partition_idx=i, tensor=mxpo_output)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=elem_type, shape=shape)
        value_info = onnx.helper.make_value_info(
            name=name,
            type_proto=tensor_type_proto
            )
        mxpo_Name.append(name)
        model.graph.value_info.insert(tensor_idx + 1 + i, value_info)
    model.graph.value_info.remove(mxpo_output)



    for i in range(partition):
        
        for attr in conv_node.attribute:
            conv_attrs = {attr.name: helper.get_attribute_value(attr) for attr in conv_node.attribute}
            # print(f"Conv attributes: {conv_attrs}")
        if len(conv_node.input) == 3:
            conv = onnx.helper.make_node(
                            'Conv',
                            name=conv_node.name + f"_{i}",
                            inputs=[conv_node.input[0], kernel_Name[i], bias_Name[i]],
                            outputs=[conv_Name[i]],
                            **conv_attrs
                            
                            ) 
        else:
            conv = onnx.helper.make_node(
                            'Conv',
                            name=conv_node.name + f"_{i}",
                            inputs=[conv_node.input[0], kernel_Name[i]],
                            outputs=[conv_Name[i]],
                            **conv_attrs
                            )
        convNameList.append(conv.name)
        model.graph.node.insert(conv_index + 1 + i, conv)
    # Create 
    model.graph.node.remove(conv_node)
    relu_index = 0
    relu_node = None
    for i, node in enumerate(model.graph.node):
        if node.name == relu_node_name:
            relu_node = node
            relu_index = i
            
    for i in range(partition):
        relu = onnx.helper.make_node(
                        'Relu',
                        name=relu_node.name + f"_{i}",
                        inputs=[conv_Name[i]],
                        outputs=[relu_Name[i]],
                        ) 
        model.graph.node.insert(relu_index + 1 + i, relu)
    model.graph.node.remove(relu_node)

    # MaxPool
    mxpo_node =None
    mxpo_index = i
    for i, node in enumerate(model.graph.node):
        if node.name == mxpo_node_name:
            mxpo_node = node
            mxpo_index = i

    for i in range(partition):
        
        for attr in mxpo_node.attribute:
            mxpo_attrs = {attr.name: helper.get_attribute_value(attr) for attr in mxpo_node.attribute}
        maxpool = onnx.helper.make_node(
                        'MaxPool',
                        name=mxpo_node.name + f"_{i}",
                        inputs=[relu_Name[i]],
                        outputs=[mxpo_Name[i]],
                        **mxpo_attrs
                        ) 
        maxPoolNameList.append(maxpool.name)
        model.graph.node.insert(mxpo_index + 1 + i, maxpool)
    model.graph.node.remove(mxpo_node)

    # Concat
    concat = onnx.helper.make_node(
                    'Concat',
                    name=concat_node_name,
                    inputs=mxpo_Name,
                    outputs=mxpo_node.output,
                    axis=1
                    ) 
    model.graph.node.insert(mxpo_index + partition , concat)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    

    return model, convNameList, maxPoolNameList
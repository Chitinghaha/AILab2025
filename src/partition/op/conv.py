from . import *
def partition_output_channel(partition:int, model:Union[onnx.ModelProto, str], convNode:Union[onnx.NodeProto,str]) -> onnx.ModelProto:
        
    w_NameList = []
    b_NameList = []
    o_NameList = []
    convNodeNameList = []
    if isinstance(convNode, str): _, convNode = util.get_node_Info(convNode,model=model)

    # Weight : TensorInfoProto
    w_idx, weight = util.get_initilizer(convNode.input[1],model)
    w_data = numpy_helper.to_array(weight)
    number = util.output_shape(partition_num=partition,shape=w_data.shape[0])
    for p_idx in range(partition):
        dims, vals = util.partition_numpy_kernel(number=number, partition_idx=p_idx, tensor=w_data)
        name = weight.name + f"/split_{p_idx}"
        tensor = onnx.helper.make_tensor(
            name=name, data_type=weight.data_type,
            dims=dims, vals=vals)
        model.graph.initializer.insert(w_idx + 1 + p_idx, tensor)
        w_NameList.append(name)
    model.graph.initializer.remove(weight)
    
    # Bias : TensorInfoProto
    b_idx, bias = -1, None
    if len(convNode.input) == 3:
        b_idx, bias = util.get_initilizer(convNode.input[2],model)
        b_data =  numpy_helper.to_array(bias)
        number = util.output_shape(partition_num=partition,shape=b_data.shape[0])
        for p_idx in range(partition):
            name = bias.name + f"/split_{p_idx}"
            minn = sum(number[0:p_idx])
            maxx = sum(number[0:p_idx+1])
            tensor = onnx.helper.make_tensor(
                name=name, data_type=bias.data_type,
                dims=[maxx-minn], vals=b_data[minn:maxx])
            model.graph.initializer.insert(b_idx + 1 + p_idx, tensor)
            b_NameList.append(name)
        model.graph.initializer.remove(bias)
    # Conv Output : ValueInfoProto
    opt_idx, conv_opt = util.get_value_info(tensor_name=convNode.output[0],model=model)
    for p_idx in range(partition):
        name = conv_opt.name + f"/split_{p_idx}"
        elem_type, shape = util.partition_output(number=number, partition_idx=p_idx, tensor=conv_opt)
        value_info = onnx.helper.make_tensor_value_info(name=name,elem_type=elem_type, shape=shape)
        model.graph.value_info.insert(opt_idx + 1 + p_idx, value_info)
        o_NameList.append(name)
    model.graph.value_info.remove(conv_opt)
    # Create Conv Node : NodeProto
    c_idx, convNode = util.get_node_Info(convNode.name, model)
    for p_idx in range(partition):
        if bias == None:
            inputsList = [convNode.input[0], w_NameList[p_idx]]
        else:
            inputsList = [convNode.input[0], w_NameList[p_idx], b_NameList[p_idx]]
        conv = onnx.helper.make_node(
                        'Conv',
                        name=convNode.name + f"_{p_idx}",
                        inputs=inputsList,
                        outputs=[o_NameList[p_idx]],
                        dilations=[convNode.attribute[0].ints[0],convNode.attribute[0].ints[1]],
                        group=convNode.attribute[1].i,
                        kernel_shape=[convNode.attribute[2].ints[0],convNode.attribute[2].ints[1]],
                        pads=[convNode.attribute[3].ints[0],convNode.attribute[3].ints[1],convNode.attribute[3].ints[2],convNode.attribute[3].ints[3]],
                        strides=[convNode.attribute[4].ints[0],convNode.attribute[4].ints[1]],
                        ) 
        convNodeNameList.append(conv.name)
        model.graph.node.insert(c_idx + 1 + p_idx, conv)
    model.graph.node.remove(convNode)
    return model, convNodeNameList

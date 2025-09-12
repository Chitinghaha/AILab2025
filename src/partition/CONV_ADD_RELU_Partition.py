from . import *
from src.partition.op import conv


def CONV_ADD_RELU_Partition(partition:int, 
                            model:onnx.ModelProto, 
                            conv_node_name:str, 
                            add_node_name:str, 
                            relu_node_name:str, 
                            split_node_name:str,
                            concat_node_name:str
                            ) -> tuple([onnx.ModelProto, list, str, list]):
    _, convNode = util.get_node_Info(model=model, node_name=conv_node_name)
    ipt = [tensor.name for tensor in model.graph.input]
    ipt.remove(convNode.input[0])
    iptAdd_NameList = []
    a_idx, addNode =  util.get_node_Info(model=model, node_name=add_node_name)
    model, convNodeNameList = conv.partition_output_channel(partition=partition,model=model,convNode=conv_node_name)
    ipt_add_idx, add_ipt = util.get_value_info(tensor_name=ipt[0],model=model)
    _, dims = util.get_valueInfo_dim_type(add_ipt, model)
    number = util.output_shape(partition_num=partition,shape=dims[1])
    # Split Output - Add Input
    for p_idx in range(partition):
        name = add_ipt.name + f"/split_{p_idx}_"
        elem_type, shape = util.partition_output(number=number, partition_idx=p_idx, tensor=add_ipt)
        value_info = onnx.helper.make_tensor_value_info(name=name,elem_type=elem_type, shape=shape)
        model.graph.value_info.insert(ipt_add_idx + 1 + p_idx, value_info)
        iptAdd_NameList.append(name)
    # Split Operator
    split = onnx.helper.make_node(
            op_type='Split',
            name=split_node_name,
            inputs=ipt,
            outputs=iptAdd_NameList,
            axis=1
        )
    model.graph.node.insert(a_idx, split)
    
    # Add
    a_idx, addNode =  util.get_node_Info(model=model, node_name=add_node_name)
    a_opt_idx, addOptTensor = util.get_value_info(tensor_name=addNode.output[0],model=model)
    addOptNameList = []
    addNodeNameList = []
    for p_idx, [ipt1TensorName, ipt2NodeName] in enumerate(zip(iptAdd_NameList, convNodeNameList)):
        _, ipt2Node = util.get_node_Info(model=model, node_name=ipt2NodeName)
        ipt2TensorName = ipt2Node.output[0]
        _, ipt2Tensor = util.get_value_info(model=model, tensor_name=ipt2TensorName)
        name = addNode.output[0] + f"/split_{p_idx}"
        elem_type, shape = util.get_valueInfo_dim_type(ipt2Tensor, model)
        value_info = onnx.helper.make_tensor_value_info(
            name=name,elem_type=elem_type, shape=shape)
        model.graph.value_info.insert(a_opt_idx + 1 + p_idx, value_info)
        addOptNameList.append(name)
        
        add = onnx.helper.make_node(
                'Add',
                name=addNode.name + f"_{p_idx}",
                inputs=[ipt1TensorName, ipt2TensorName],
                outputs=[name],
                )
        addNodeNameList.append(add.name)
        model.graph.node.insert(a_idx+1+p_idx, add)
    # ReLU - Create Output Tensor & Create ReLU Operator
    r_idx, reluNode = util.get_node_Info(node_name=relu_node_name,model=model)
    r_opt_idx, reluOptTensor = util.get_value_info(tensor_name=reluNode.output[0],model=model)
    reluOptNameList = []
    reluNodeNameList = []
    for p_idx, ipt in enumerate(addOptNameList):
        name = reluNode.output[0] + f"/concat_{p_idx}"
        elem_type, shape = util.get_valueInfo_dim_type(ipt, model)
        value_info = onnx.helper.make_tensor_value_info(
            name=name,elem_type=elem_type, shape=shape)
        model.graph.value_info.insert(r_opt_idx + 1 + p_idx, value_info)
        reluOptNameList.append(name)
        relu = onnx.helper.make_node(
                'Relu',
                name=reluNode.name + f"_{p_idx}",
                inputs=[ipt],
                outputs=[name],
                )
        model.graph.node.insert(r_idx + 1 + p_idx, relu)
        reluNodeNameList.append(reluNode.name + f"_{p_idx}")
    
    idx, _ = util.get_node_Info(node_name=reluNodeNameList[-1],model=model)
    # Concat
    concat = onnx.helper.make_node(
                    'Concat',
                    name=concat_node_name,
                    inputs=reluOptNameList,
                    outputs=reluNode.output,
                    axis=1
                    ) 
    model.graph.node.insert(idx + 1 , concat)
    model.graph.value_info.remove(addOptTensor)
    model.graph.node.remove(addNode)
    model.graph.node.remove(reluNode)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model, convNodeNameList, addNodeNameList, reluNodeNameList

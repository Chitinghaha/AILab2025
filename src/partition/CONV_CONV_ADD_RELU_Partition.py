from . import *
from src.partition.op import conv
def CONV_CONV_ADD_RELU_Partition(partition:int, 
                                 model:onnx.ModelProto, 
                                 conv_node_name_1:str, 
                                 conv_node_name_2:str,
                                 add_node_name:str, 
                                 relu_node_name:str, 
                                 concat_node_name:str) -> onnx.ModelProto:

    model, convNodeNameList_1 = conv.partition_output_channel(partition=partition,model=model,convNode=conv_node_name_1)
    model, convNodeNameList_2 = conv.partition_output_channel(partition=partition,model=model,convNode=conv_node_name_2)

    # Add - Create Output Tensor & Create Add Operator
    a_idx, addNode = util.get_node_Info(node_name=add_node_name,model=model)
    a_opt_idx, addOptTensor = util.get_value_info(tensor_name=addNode.output[0],model=model)
    addOptNameList = []
    for p_idx, [ipt1, ipt2] in enumerate(zip(convNodeNameList_1, convNodeNameList_2)):
        _, convNode1 = util.get_node_Info(ipt1, model)
        _, convNode2 = util.get_node_Info(ipt2, model)
        
        name = addNode.output[0] + f"/split_{p_idx}"
        elem_type, shape = util.get_valueInfo_dim_type(convNode1.output[0], model)
        value_info = onnx.helper.make_tensor_value_info(
            name=name,elem_type=elem_type, shape=shape)
        model.graph.value_info.insert(a_opt_idx + 1 + p_idx, value_info)
        addOptNameList.append(name)
        
        add = onnx.helper.make_node(
                'Add',
                name=addNode.name + f"_{p_idx}",
                inputs=[convNode1.output[0], convNode2.output[0]],
                outputs=[name],
                )
        model.graph.node.insert(a_idx + 1 + p_idx, add)

    
    # ReLU - Create Output Tensor & Create ReLU Operator
    r_idx, reluNode = util.get_node_Info(node_name=relu_node_name,model=model)
    r_opt_idx, reluOptTensor = util.get_value_info(tensor_name=reluNode.output[0],model=model)
    reluOptNameList = []
    reluNodeNameList = []
    for p_idx, ipt in enumerate(addOptNameList):
        name = reluNode.output[0] + f"/split_{p_idx}"
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
    return model, list(map(list, zip(convNodeNameList_1, convNodeNameList_2))), reluNodeNameList

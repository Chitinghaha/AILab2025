import onnx
from src.scheduler.scheduler import scheduler
from src.tool import util

import argparse
parser = argparse.ArgumentParser()


model = onnx.load("out/googlenet-v12_no_dropout/googlenet_subgraph.onnx")
nodeList, nodeDict = util.create_operator_list_dict(model,{})
scheduling = scheduler(nodeDict, nodeList)




for i in scheduling:
    n_idx, PEnode = util.get_node_Info(node_name=str(i), model=model)
    submodel = onnx.load(f"out/googlenet-v12_no_dropout/subgraph/googlenet_subgraph_Node_{nodeList[i]}.onnx")
    v_idx, _ = util.get_value_info(tensor_name=PEnode.input[-1], model=model)
    
    for idx, value_info in enumerate(submodel.graph.value_info):
        model.graph.value_info.insert(v_idx+idx+1,value_info)
    for idx, node in enumerate(submodel.graph.node):
        model.graph.node.insert(n_idx+1+idx,node)
    
    subnodeList, subnodeDict = util.create_operator_list_dict(submodel,{})
    for index in subnodeList:
        _, node = util.get_node_Info(index,submodel)
    model.graph.node.remove(PEnode)   
onnx.save(model,f"out/googlenet-v12_no_dropout/googlenet_subgraph_transform.onnx")


# python3 googlenet_subgraph_transform.py
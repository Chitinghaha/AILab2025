import sys
import argparse
parser = argparse.ArgumentParser()
from src.tool import util
from src.partition.CONV_RELU_MAXPOOL_Partition import CONV_RELU_MAXPOOL_Partition
from src.partition.CONV_RELU_Partition import CONV_RELU_Partition
from src.partition.GAP_FLATTEN_GEMM_Patrition import GAP_FLATTEN_GEMM_Patrition
from src.partition.CONV_CONV_ADD_RELU_Partition import CONV_CONV_ADD_RELU_Partition
from src.partition.CONV_ADD_RELU_Partition import CONV_ADD_RELU_Partition
import os
import onnx





parser.add_argument('-m','--modelpath', required=True, help='onnx model path')
parser.add_argument('-c','--core', required=True, help='Number of core')
parser.add_argument('-o','--subgraph', required=True, help='onnx submodel folder path')
args = parser.parse_args()

dirPath, _ = os.path.splitext(args.modelpath)
modelname = os.path.basename(dirPath)

if not os.path.exists(os.path.join(args.subgraph, "spatial")):
    os.makedirs(os.path.join(args.subgraph, "spatial"))


model = onnx.load(args.modelpath)


nodeList = [ node.op_type for node in model.graph.node]
nodeName = [ node.name for node in model.graph.node]
if not os.path.exists(os.path.join(args.subgraph, 'spatial')):
    os.makedirs(os.path.join(args.subgraph, 'spatial'))
concat_node_name = f"/concat/concat/{modelname}"
split_node_name = f"/split/split/{modelname}"
sum_node_name = f"/sum/sum/{modelname}"
subgraphList = []

if nodeList == ['Conv', 'Relu', 'MaxPool']:
    model, convNameList, maxPoolNameList  = CONV_RELU_MAXPOOL_Partition(
                        partition=int(args.core),
                        model=model,
                        conv_node_name=nodeName[0],
                        relu_node_name=nodeName[1],
                        mxpo_node_name=nodeName[2],
                        concat_node_name=concat_node_name) 
    for idx, [conv, maxPool] in enumerate(zip(convNameList,maxPoolNameList)):
        submodel = util.extract_subgraph_node2node(model,firstNode=conv,finalNode=maxPool)
        subgraphList.append(submodel)
        onnx.save(submodel,os.path.join(args.subgraph, 'spatial', f"Node_{modelname}_PE_{idx}.onnx"))
    # create Graph Node_{nodeIdx} to Node_{nodeIdx}_PE.onnx
    iptList = []
    valueInfoList = []
    optList = []
    concatIpt = []
    nodeList = []
    # Input tensors (ValueInfoProto).
    for idx, subModel in enumerate(subgraphList): 
        for ipt in subModel.graph.input:
            _, X = util.get_value_info(ipt.name,subModel)
            if X not in iptList:
                iptList.append(X)
        for opt in subModel.graph.output:
            _, X = util.get_value_info(opt.name,subModel)   
            if X not in valueInfoList:
                valueInfoList.append(X)
        # First Node (NodeInfoProto)
        node = onnx.helper.make_node(
                op_type=f"PE{idx}",
                inputs=[ipt.name for ipt in subModel.graph.input],
                outputs=[opt.name for opt in subModel.graph.output],
                name=f"Node_{modelname}_PE_{idx}"
            )
        nodeList.append(node)
    # Insert Concat Operator
    _, concat = util.get_node_Info(concat_node_name, model)
    for opt in concat.output:
        _, tensor = util.get_value_info(opt, model)
        optList.append(tensor)
    nodeList.append(concat)
    
elif nodeList == ['Conv', 'Relu']:
    model, convNameList, reluNameList = CONV_RELU_Partition(
                        partition=int(args.core),
                        model=model,
                        conv_node_name=nodeName[0],
                        relu_node_name=nodeName[1],
                        concat_node_name=concat_node_name) 
    for idx, [conv, relu] in enumerate(zip(convNameList,reluNameList)):
        submodel = util.extract_subgraph_node2node(model,firstNode=conv,finalNode=relu)
        subgraphList.append(submodel)
        onnx.save(submodel,os.path.join(args.subgraph, 'spatial', f"Node_{modelname}_PE_{idx}.onnx"))
    # create Graph Node_{nodeIdx} to Node_{nodeIdx}_PE.onnx
    iptList = []
    valueInfoList = []
    optList = []
    concatIpt = []
    nodeList = []
    # Input tensors (ValueInfoProto).
    for idx, subModel in enumerate(subgraphList): 
        for ipt in subModel.graph.input:
            _, X = util.get_value_info(ipt.name,subModel)
            if X not in iptList:
                iptList.append(X)
        for opt in subModel.graph.output:
            _, X = util.get_value_info(opt.name,subModel)   
            if X not in valueInfoList:
                valueInfoList.append(X)
        # First Node (NodeInfoProto)
        node = onnx.helper.make_node(
                op_type=f"PE{idx}",
                inputs=[ipt.name for ipt in subModel.graph.input],
                outputs=[opt.name for opt in subModel.graph.output],
                name=f"Node_{modelname}_PE_{idx}"
            )
        nodeList.append(node)
    # Insert Concat Operator
    _, concat = util.get_node_Info(concat_node_name, model)
    for opt in concat.output:
        _, tensor = util.get_value_info(opt, model)
        optList.append(tensor)
    nodeList.append(concat)

elif nodeList == ['GlobalAveragePool', 'Flatten', 'Gemm']:    
    model, gaplNameList, gemmNameList = GAP_FLATTEN_GEMM_Patrition(
                        partition=int(args.core),
                        model=model,
                        gapl_node_name=nodeName[0],
                        fltn_node_name=nodeName[1],
                        gemm_node_name=nodeName[2],
                        splt_node_name=split_node_name,
                        sums_node_name=sum_node_name,
                        ) 
    for idx, [gapl, gemm] in enumerate(zip(gaplNameList,gemmNameList)):
        submodel = util.extract_subgraph_node2node(model,firstNode=gapl,finalNode=gemm)
        subgraphList.append(submodel)
        onnx.save(submodel,os.path.join(args.subgraph, 'spatial', f"Node_{modelname}_PE_{idx}.onnx"))
    # create Graph Node_{nodeIdx} to Node_{nodeIdx}_PE.onnx
    valueInfoList = []
    optList = []
    concatIpt = []
    nodeList = []
    iptList  = []
    # Insert Split Operator - Input tensors (ValueInfoProto).
    _, split = util.get_node_Info(split_node_name, model)
    for ipt in split.input:
        _, tensor = util.get_value_info(ipt, model)
        iptList.append(tensor)
    nodeList.append(split)
    # Create PE subgraph
    for idx, subModel in enumerate(subgraphList): 
        for ipt in subModel.graph.input:
            _, X = util.get_value_info(ipt.name,subModel)
            if X not in valueInfoList:
                valueInfoList.append(X)
        for opt in subModel.graph.output:
            _, X = util.get_value_info(opt.name,subModel)   
            if X not in valueInfoList:
                valueInfoList.append(X)
        # First Node (NodeInfoProto)
        node = onnx.helper.make_node(
                op_type=f"PE{idx}",
                inputs=[ipt.name for ipt in subModel.graph.input],
                outputs=[opt.name for opt in subModel.graph.output],
                name=f"Node_{modelname}_PE_{idx}"
            )
        nodeList.append(node)
    # Insert Sum Operator - Output tensors (ValueInfoProto).
    _, sum = util.get_node_Info(sum_node_name, model)
    for opt in sum.output:
        _, tensor = util.get_value_info(opt, model)
        optList.append(tensor)
    nodeList.append(sum)

elif nodeList == ['Conv', 'Conv', 'Add','Relu']:
    model, convNameList, reluNameList = CONV_CONV_ADD_RELU_Partition(
                        partition=int(args.core),
                        model=model,
                        conv_node_name_1=nodeName[0],
                        conv_node_name_2=nodeName[1],
                        add_node_name=nodeName[2],
                        relu_node_name=nodeName[3],
                        concat_node_name=concat_node_name,
                        )
    for idx, [conv, relu] in enumerate(zip(convNameList,reluNameList)):
        submodel = util.extract_subgraph_multi_node2node(model,topNodes=conv,btmNodes=[relu])
        subgraphList.append(submodel)
        onnx.save(submodel,os.path.join(args.subgraph, 'spatial', f"Node_{modelname}_PE_{idx}.onnx"))
    # create Graph Node_{nodeIdx} to Node_{nodeIdx}_PE.onnx
    iptList = []
    valueInfoList = []
    optList = []
    concatIpt = []
    nodeList = []
    # Input tensors (ValueInfoProto).
    for idx, subModel in enumerate(subgraphList):
        for ipt in subModel.graph.input:
            _, X = util.get_value_info(ipt.name,subModel)
            if X not in iptList:
                iptList.append(X)
        for opt in subModel.graph.output:
            _, X = util.get_value_info(opt.name,subModel)   
            if X not in valueInfoList:
                valueInfoList.append(X)
        # First Node (NodeInfoProto)
        node = onnx.helper.make_node(
                op_type=f"PE{idx}",
                inputs=[ipt.name for ipt in subModel.graph.input],
                outputs=[opt.name for opt in subModel.graph.output],
                name=f"Node_{modelname}_PE_{idx}"
            )
        nodeList.append(node)
    # Insert Concat Operator
    _, concat = util.get_node_Info(concat_node_name, model)
    for opt in concat.output:
        _, tensor = util.get_value_info(opt, model)
        optList.append(tensor)
    nodeList.append(concat) 
elif nodeList == ['Conv', 'Add', 'Relu']:
    model, convNodeNameList, addNodeNameList, reluNodeNameList = CONV_ADD_RELU_Partition(
                        partition=int(args.core),
                        model=model,
                        conv_node_name=nodeName[0],
                        add_node_name=nodeName[1],
                        relu_node_name=nodeName[2],
                        concat_node_name=concat_node_name,
                        split_node_name=split_node_name,
                        )
    for idx, [conv_add, relu] in enumerate(zip(list(map(list, zip(convNodeNameList, addNodeNameList))), reluNodeNameList)):
        submodel = util.extract_subgraph_multi_node2node(model,topNodes=conv_add,btmNodes=[relu])
        subgraphList.append(submodel)
        onnx.save(submodel,os.path.join(args.subgraph, 'spatial', f"Node_{modelname}_PE_{idx}.onnx"))
    # create Graph Node_{nodeIdx} to Node_{nodeIdx}_PE.onnx
    valueInfoList = []
    optList = []
    concatIpt = []
    nodeList = []
    iptList  = []
    # Insert Split Operator - Input tensors (ValueInfoProto).
    _, split = util.get_node_Info(split_node_name, model)
    for ipt in split.input:
        _, tensor = util.get_value_info(ipt, model)
        iptList.append(tensor)
    nodeList.append(split)
    # Create PE subgraph
    _, node = util.get_node_Info(convNodeNameList[0], model)
    _, tensor = util.get_value_info(node.input[0], model)
    iptList.append(tensor)    
    for idx, subModel in enumerate(subgraphList): 
        for ipt in subModel.graph.input:
            _, X = util.get_value_info(ipt.name,subModel)

            if X not in valueInfoList:
                valueInfoList.append(X)
            
        for opt in subModel.graph.output:
            _, X = util.get_value_info(opt.name,subModel)   
            if X not in valueInfoList:
                valueInfoList.append(X)
        # First Node (NodeInfoProto)
        node = onnx.helper.make_node(
                op_type=f"PE{idx}",
                inputs=[ipt.name for ipt in subModel.graph.input],
                outputs=[opt.name for opt in subModel.graph.output],
                name=f"Node_{modelname}_PE_{idx}"
            )
        nodeList.append(node)
    # Insert concat Operator - Output tensors (ValueInfoProto).
    _, concat = util.get_node_Info(concat_node_name, model)
    for opt in concat.output:
        _, tensor = util.get_value_info(opt, model)
        optList.append(tensor)
    nodeList.append(concat)    


# Create the graph (GraphProto)
graph_def = onnx.helper.make_graph(
    nodes=nodeList,
    name=f"resnet50_spactial_{modelname}",
    inputs=iptList,   # Graph input
    outputs=optList,  # Graph output
    value_info=valueInfoList,
) 
# Create Model
model_def = onnx.helper.make_model(graph_def, producer_name="acai-lab16")
model_def.opset_import[0].version = 13
onnx.save(model_def, os.path.join(args.subgraph,f"resnet50_subgraph_Node_{modelname}.onnx"))



import argparse
parser = argparse.ArgumentParser()
from src.tool import util
import os
import onnx
parser.add_argument('-m','--modelpath', required=True, help='onnx model path')
parser.add_argument('-o','--subgraph', required=True, help='onnx subgraph model folder path')
args = parser.parse_args()

subgraphList = [ 
    #0
    [ "/conv1/Conv" , "/maxpool/MaxPool"],                           
    #1   
    [ "/layer1/layer1.0/conv1/Conv" , "/layer1/layer1.0/relu/Relu"],
    #2
    ["/layer1/layer1.0/conv2/Conv", "/layer1/layer1.0/relu_1/Relu"],
    #3
    [["/layer1/layer1.0/conv3/Conv", "/layer1/layer1.0/downsample/downsample.0/Conv" ] , ["/layer1/layer1.0/relu_2/Relu"]],  
    #4
    ["/layer1/layer1.1/conv1/Conv", "/layer1/layer1.1/relu/Relu"],
    #5
    ["/layer1/layer1.1/conv2/Conv", "/layer1/layer1.1/relu_1/Relu"],
    #6
    [["/layer1/layer1.1/conv3/Conv", "/layer1/layer1.1/Add"], ["/layer1/layer1.1/relu_2/Relu"]],
    #7
    ["/layer1/layer1.2/conv1/Conv", "/layer1/layer1.2/relu/Relu"],
    #8
    ["/layer1/layer1.2/conv2/Conv", "/layer1/layer1.2/relu_1/Relu"],
    #9
    [["/layer1/layer1.2/conv3/Conv", "/layer1/layer1.2/Add"], ["/layer1/layer1.2/relu_2/Relu"]],
    #10    
    ["/layer2/layer2.0/conv1/Conv", "/layer2/layer2.0/relu/Relu"],
    #11
    ['/layer2/layer2.0/conv2/Conv',"/layer2/layer2.0/relu_1/Relu"],
    #12
    [["/layer2/layer2.0/conv3/Conv", "/layer2/layer2.0/downsample/downsample.0/Conv"], ["/layer2/layer2.0/relu_2/Relu"]],
    #13
    ["/layer2/layer2.1/conv1/Conv", "/layer2/layer2.1/relu/Relu"],
    #14
    ["/layer2/layer2.1/conv2/Conv", "/layer2/layer2.1/relu_1/Relu"],
    #15
    [["/layer2/layer2.1/conv3/Conv", "/layer2/layer2.1/Add"], ["/layer2/layer2.1/relu_2/Relu"]],
    #16
    ["/layer2/layer2.2/conv1/Conv", "/layer2/layer2.2/relu/Relu"],
    #17
    ["/layer2/layer2.2/conv2/Conv", "/layer2/layer2.2/relu_1/Relu"],
    #18
    [["/layer2/layer2.2/conv3/Conv", "/layer2/layer2.2/Add"], ["/layer2/layer2.2/relu_2/Relu"]],
    #19
    ["/layer2/layer2.3/conv1/Conv", "/layer2/layer2.3/relu/Relu"],
    #20
    ["/layer2/layer2.3/conv2/Conv", "/layer2/layer2.3/relu_1/Relu"],
    #21
    [["/layer2/layer2.3/conv3/Conv", "/layer2/layer2.3/Add"],["/layer2/layer2.3/relu_2/Relu"]],
    #22
    ["/layer3/layer3.0/conv1/Conv", "/layer3/layer3.0/relu/Relu"],
    #23
    ["/layer3/layer3.0/conv2/Conv", "/layer3/layer3.0/relu_1/Relu"],
    #24
    [["/layer3/layer3.0/conv3/Conv","/layer3/layer3.0/downsample/downsample.0/Conv"],["/layer3/layer3.0/relu_2/Relu"]],
    #25
    ["/layer3/layer3.1/conv1/Conv", "/layer3/layer3.1/relu/Relu"],
    #26
    ["/layer3/layer3.1/conv2/Conv", "/layer3/layer3.1/relu_1/Relu"],
    #27
    [["/layer3/layer3.1/conv3/Conv", "/layer3/layer3.1/Add"],["/layer3/layer3.1/relu_2/Relu"]],
    #28
    ["/layer3/layer3.2/conv1/Conv", "/layer3/layer3.2/relu/Relu"],
    #29
    ["/layer3/layer3.2/conv2/Conv", "/layer3/layer3.2/relu_1/Relu"],
    #30
    [["/layer3/layer3.2/conv3/Conv", "/layer3/layer3.2/Add"], ["/layer3/layer3.2/relu_2/Relu"]],
    #31
    ["/layer3/layer3.3/conv1/Conv", "/layer3/layer3.3/relu/Relu"],
    #32
    ["/layer3/layer3.3/conv2/Conv", "/layer3/layer3.3/relu_1/Relu"],
    #33
    [["/layer3/layer3.3/conv3/Conv","/layer3/layer3.3/Add"], ["/layer3/layer3.3/relu_2/Relu"]],
    #34
    ["/layer3/layer3.4/conv1/Conv", "/layer3/layer3.4/relu/Relu"],
    #35
    ["/layer3/layer3.4/conv2/Conv", "/layer3/layer3.4/relu_1/Relu"],
    #36
    [["/layer3/layer3.4/conv3/Conv", "/layer3/layer3.4/Add"], ["/layer3/layer3.4/relu_2/Relu"]],
    #37
    ["/layer3/layer3.5/conv1/Conv", "/layer3/layer3.5/relu/Relu"],
    #38
    ["/layer3/layer3.5/conv2/Conv", "/layer3/layer3.5/relu_1/Relu"],
    #39
    [["/layer3/layer3.5/conv3/Conv", "/layer3/layer3.5/Add"], ["/layer3/layer3.5/relu_2/Relu"]],
    #40
    ["/layer4/layer4.0/conv1/Conv", "/layer4/layer4.0/relu/Relu"],
    #41
    ["/layer4/layer4.0/conv2/Conv", "/layer4/layer4.0/relu_1/Relu"],
    #42
    [["/layer4/layer4.0/conv3/Conv", "/layer4/layer4.0/downsample/downsample.0/Conv"],["/layer4/layer4.0/relu_2/Relu"]],
    #43
    ["/layer4/layer4.1/conv1/Conv", "/layer4/layer4.1/relu/Relu"],
    #44
    ["/layer4/layer4.1/conv2/Conv", "/layer4/layer4.1/relu_1/Relu"],
    #45
    [["/layer4/layer4.1/conv3/Conv", "/layer4/layer4.1/Add"], ["/layer4/layer4.1/relu_2/Relu"]],
    #46
    ["/layer4/layer4.2/conv1/Conv", "/layer4/layer4.2/relu/Relu"],
    #47
    ["/layer4/layer4.2/conv2/Conv", "/layer4/layer4.2/relu_1/Relu"],
    #48
    [["/layer4/layer4.2/conv3/Conv", "/layer4/layer4.2/Add"], ["/layer4/layer4.2/relu_2/Relu"]],
    #49
    ["/avgpool/GlobalAveragePool", "/fc/Gemm"]
]


adjList = [
    [1,3],  # 0
    [2],    # 1
    [3],    # 2
    [4,6],  # 3
    [5],    # 4
    [6],    # 5
    [7,9],  # 6
    [8],    # 7
    [9],    # 8
    [10,12],# 9
    [11],   # 10
    [12],   # 11
    [13,15],# 12
    [14],   # 13
    [15],   # 14
    [16,18],# 15
    [17],   # 16
    [18],   # 17
    [19,21],# 18
    [20],   # 19
    [21],   # 20
    [22,24],# 21
    [23],   # 22
    [24],   # 23
    [25,27],# 24
    [26],   # 25
    [27],   # 26
    [28,30],# 27
    [29],   # 28
    [30],   # 29
    [31,33],# 30
    [32],   # 31
    [33],   # 32
    [34,36],# 33
    [35],   # 34
    [36],   # 35
    [37,39],# 36
    [38],   # 37
    [39],   # 38
    [40,42],# 39
    [41],   # 40
    [42],   # 41
    [43,45],# 42
    [44],   # 43
    [45],   # 44
    [46,48],# 45
    [47],   # 46
    [48],   # 47
    [49],   # 48
    [  ],   # 49
    ]

if not os.path.exists(os.path.join(args.subgraph, "subgraph","temporal")):
    os.makedirs(os.path.join(args.subgraph, "subgraph","temporal"))

dirPath, _ = os.path.splitext(args.modelpath)
modelname = os.path.basename(dirPath)

model = onnx.load(args.modelpath)

for idx in range(len(subgraphList)):
    first, final = subgraphList[idx]
    if isinstance(first, list) and isinstance(final, list):
        subModel = util.extract_subgraph_multi_node2node(model=model, topNodes=first, btmNodes=final)
    elif isinstance(first, list) and isinstance(final, str):
        subModel = util.extract_subgraph_multi_node2node(model=model, topNodes=first, btmNodes=[final])
    elif isinstance(first, str) and isinstance(final, list):
        subModel = util.extract_subgraph_multi_node2node(model=model, topNodes=[first], btmNodes=final)
    else:
        subModel = util.extract_subgraph_node2node(model=model, firstNode=first, finalNode=final)

    onnx.save(subModel, os.path.join(args.subgraph, "subgraph", "temporal" ,f"{idx}.onnx"))


iptList = []
optList = []
nodeList = []
valueInfoList = []

# Input tensors (ValueInfoProto).
subModel = onnx.load(os.path.join(args.subgraph, "subgraph" , 'temporal', f"{0}.onnx"))
for ipt in subModel.graph.input:
    _, X = util.get_value_info(ipt.name,subModel)
    iptList.append(X)
for opt in subModel.graph.output:
    _, X = util.get_value_info(opt.name,subModel)    
    valueInfoList.append(X)
# First Node (NodeInfoProto)
firstNode = onnx.helper.make_node(
        op_type="0",
        inputs=[ipt.name for ipt in subModel.graph.input],
        outputs=[opt.name for opt in subModel.graph.output],
        name="0"
    )    
nodeList.append(firstNode)



# Create activative parameter and Node (ValueInfoProto, NodeInfoProto)
for idx in range(1, len(subgraphList) - 1):
    subModel = onnx.load(os.path.join(args.subgraph, "subgraph" , 'temporal', f"{idx}.onnx"))
    for ipt in subModel.graph.input:
        _, X = util.get_value_info(ipt.name,subModel)
        valueInfoList.append(X)
    for opt in subModel.graph.output:
        _, X = util.get_value_info(opt.name,subModel)    
        valueInfoList.append(X) 
    iptTensor =  subModel.graph.input
    optTensor =  subModel.graph.output
    Node = onnx.helper.make_node(
        op_type=str(idx),
        inputs=[ipt.name for ipt in subModel.graph.input],
        outputs=[opt.name for opt in subModel.graph.output],
        name=str(idx)
    ) 
    nodeList.append(Node)


# Output tensors (ValueInfoProto).
idx = len(subgraphList) - 1
subModel = onnx.load(os.path.join(args.subgraph, "subgraph" , "temporal", f"{idx}.onnx"))
for ipt in subModel.graph.input:
    _, X = util.get_value_info(ipt.name,subModel)
    valueInfoList.append(X)
for opt in subModel.graph.output:
    _, X = util.get_value_info(opt.name,subModel)    
    optList.append(X)
# Final Node (NodeInfoProto)
finalNode = onnx.helper.make_node(
        op_type=f"{idx}",
        inputs=[ipt.name for ipt in subModel.graph.input],
        outputs=[opt.name for opt in subModel.graph.output],
        name=f"{idx}"
    )    
nodeList.append(finalNode)
    
# Create the graph (GraphProto)
graph_def = onnx.helper.make_graph(
    nodes=nodeList,
    name=f"resnet50_spatial_{modelname}",
    inputs=iptList,   # Graph input
    outputs=optList,  # Graph output
    value_info=valueInfoList,
) 

model_def = onnx.helper.make_model(graph_def, producer_name="acai-lab16")
model_def.opset_import[0].version = 13
onnx.save(model_def, os.path.join(args.subgraph,"resnet50_subgraph.onnx"))




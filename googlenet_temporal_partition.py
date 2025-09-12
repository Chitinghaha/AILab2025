import argparse
parser = argparse.ArgumentParser()
from src.tool import util
import os
import onnx
parser.add_argument('-m','--modelpath', required=True, help='onnx model path')
parser.add_argument('-o','--subgraph', required=True, help='onnx subgraph model folder path')
args = parser.parse_args()

# 方便後面的 googlenet_spatial_partition.py， 盡量用重複的 subgraphList
subgraphList = [ 
    # 0
    ["Conv_0", "MaxPool_2"],                           
    # 1
    ["LRN_3", "LRN_3"],
    # 2
    ["Conv_4", "Relu_5"],
    # 3
    ["Conv_6", "Relu_7"],
    # 4
    ["LRN_8", "LRN_8"],
    # 5
    ["MaxPool_9", "MaxPool_9"],
    # 6
    ["Conv_10", "Relu_11"],
    # 7
    ["Conv_12", "Relu_13"],
    # 8
    ["Conv_14", "Relu_15"],
    # 9
    ["Conv_16", "Relu_17"],
    # 10
    ["Conv_18", "Relu_19"],
    # 11
    ["MaxPool_20", "MaxPool_20"],
    # 12
    ["Conv_21", "Relu_22"],
    # 13
    ["Concat_23", "Concat_23"],
    # 14
    ["Conv_24", "Relu_25"],
    # 15
    ["Conv_26", "Relu_27"],
    # 16
    ["Conv_28", "Relu_29"],
    # 17
    ["Conv_30", "Relu_31"],
    # 18
    ["Conv_32", "Relu_33"],
    # 19
    ["MaxPool_34", "MaxPool_34"],
    # 20
    ["Conv_35", "Relu_36"],
    # 21
    ["Concat_37", "Concat_37"],
    # 22
    ["MaxPool_38", "MaxPool_38"],
    # 23 
    ["Conv_39", "Relu_40"],
    # 24 
    ["Conv_41", "Relu_42"],
    # 25 
    ["Conv_43", "Relu_44"],
    # 26 
    ["Conv_45", "Relu_46"],
    # 27 
    ["Conv_47", "Relu_48"],
    # 28 
    ["MaxPool_49", "MaxPool_49"],
    # 29 
    ["Conv_50", "Relu_51"],
    # 30 
    ["Concat_52", "Concat_52"],
    # 31
    ["Conv_53", "Relu_54"],
    # 32
    ["Conv_55", "Relu_56"],
    # 33
    ["Conv_57", "Relu_58"],
    # 34
    ["Conv_59", "Relu_60"],
    # 35
    ["Conv_61", "Relu_62"],
    # 36
    ["MaxPool_63", "MaxPool_63"],
    # 37
    ["Conv_64", "Relu_65"],
    # 38
    ["Concat_66", "Concat_66"],
    # 39
    ["Conv_67", "Relu_68"],
    # 40
    ["Conv_69", "Relu_70"],
    # 41
    ["Conv_71", "Relu_72"],
    # 42
    ["Conv_73", "Relu_74"],
    # 43
    ["Conv_75", "Relu_76"],
    # 44
    ["MaxPool_77", "MaxPool_77"],
    # 45
    ["Conv_78", "Relu_79"],
    # 46
    ["Concat_80", "Concat_80"],
    # 47
    ["Conv_81", "Relu_82"],
    # 48
    ["Conv_83", "Relu_84"],
    # 49
    ["Conv_85", "Relu_86"],
    # 50
    ["Conv_87", "Relu_88"],
    # 51
    ["Conv_89", "Relu_90"],
    # 52
    ["MaxPool_91", "MaxPool_91"],
    # 53
    ["Conv_92", "Relu_93"],
    # 54
    ["Concat_94", "Concat_94"],
    # 55
    ["Conv_95", "Relu_96"],
    # 56
    ["Conv_97", "Relu_98"],
    # 57
    ["Conv_99", "Relu_100"],
    # 58
    ["Conv_101", "Relu_102"],
    # 59
    ["Conv_103", "Relu_104"],
    # 60
    ["MaxPool_105", "MaxPool_105"],
    # 61
    ["Conv_106", "Relu_107"],
    # 62
    ["Concat_108", "Concat_108"],
    # 63
    ["MaxPool_109", "MaxPool_109"],
    # 64
    ["Conv_110", "Relu_111"],
    # 65
    ["Conv_112", "Relu_113"],
    # 66
    ["Conv_114", "Relu_115"],
    # 67
    ["Conv_116", "Relu_117"],
    # 68
    ["Conv_118", "Relu_119"],
    # 69
    ["MaxPool_120", "MaxPool_120"],
    # 70
    ["Conv_121", "Relu_122"],
    # 71
    ["Concat_123", "Concat_123"],
    # 72
    ["Conv_124", "Relu_125"],
    # 73
    ["Conv_126", "Relu_127"],
    # 74
    ["Conv_128", "Relu_129"],
    # 75
    ["Conv_130", "Relu_131"],
    # 76
    ["Conv_132", "Relu_133"],
    # 77
    ["MaxPool_134", "MaxPool_134"],
    # 78
    ["Conv_135", "Relu_136"],
    # 79
    ["Concat_137", "Concat_137"],
    # 80
    ["AveragePool_138", "Gemm_141"],
    # 81
    ["Softmax_142", "Softmax_142"]
]

adjList = [
    [1],       # 0
    [2],       # 1
    [3],       # 2
    [4],       # 3
    [5],       # 4
    [6,7,9,11],# 5
    [13],      # 6
    [8],       # 7
    [13],      # 8
    [10],      # 9
    [13],      #10
    [12],      #11
    [13],      #12
    [14,15,17,19], #13
    [21],      #14
    [16],      #15
    [21],      #16
    [18],      #17
    [21],      #18
    [20],      #19
    [21],      #20
    [22],      #21
    [23,24,26,28],#22
    [30],      #23
    [25],      #24
    [30],      #25
    [27],      #26
    [30],      #27
    [29],      #28
    [30],      #29
    [31,32,34,36],#30
    [38],      #31
    [33],      #32
    [38],      #33
    [35],      #34
    [38],      #35
    [37],      #36
    [38],      #37
    [39,40,42,44],#38
    [46],      #39
    [41],      #40
    [46],      #41
    [43],      #42
    [46],      #43
    [45],      #44
    [46],      #45
    [47,48,50,52],#46
    [54],      #47
    [49],      #48
    [54],      #49
    [51],      #50
    [54],      #51
    [53],      #52
    [54],      #53
    [55,56,58,60],#54
    [62],      #55
    [57],      #56
    [62],      #57
    [59],      #58
    [62],      #59
    [61],      #60
    [62],      #61
    [63],      #62
    [64,65,67,69],#63
    [71],      #64
    [66],      #65
    [71],      #66
    [68],      #67
    [71],      #68
    [70],      #69
    [71],      #70
    [72,73,75,77],#71
    [79],      #72
    [74],      #73
    [79],      #74
    [76],      #75
    [79],      #76
    [78],      #77
    [79],      #78
    [80],      #79
    [81],      #80
    []         #81
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


# add to spatial
# Create the graph (GraphProto)
graph_def = onnx.helper.make_graph(
    nodes=nodeList,
    name=f"googlenet_spatial_{modelname}",
    inputs=iptList,   # Graph input
    outputs=optList,  # Graph output
    value_info=valueInfoList,
) 

model_def = onnx.helper.make_model(graph_def, producer_name="acai-lab16")
model_def.opset_import[0].version = 13
onnx.save(model_def, os.path.join(args.subgraph,"googlenet_subgraph.onnx"))

# python3 googlenet_temporal_partition.py -m model/format-v7/googlenet-v12_no_dropout.onnx -o out/googlenet-v12_no_dropout


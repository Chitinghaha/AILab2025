import onnx
from src.scheduler.scheduler import scheduler
from src.tool import util
import json

partition = 4

model = onnx.load("out/googlenet-v12_no_dropout/googlenet_subgraph.onnx")
nodeList, nodeDict = util.create_operator_list_dict(model,{})
scheduling = scheduler(nodeDict, nodeList)

iptNameList = [ipt.name for ipt in model.graph.input]
iptList = model.graph.input

optNameList = [opt.name for opt in model.graph.output]
optList = model.graph.output
for pivot in range(partition):
    vluList = []
    nodList = []
    PEList = ['PE0','PE1','PE2','PE3']

    PEList.remove(f"PE{pivot}")
    PEDict = {}
    
    for i in scheduling:
        pe_node = []
        other_node = []
        submodel = onnx.load(f"out/googlenet-v12_no_dropout/subgraph/googlenet_subgraph_Node_{nodeList[i]}.onnx")
        for idx, node in enumerate(submodel.graph.node):
            if node.op_type not in PEList:
                nodList.append(node)
                for ipt in node.input:
                    _, tensor = util.get_value_info(ipt, submodel)
                    vluList.append(tensor)
                for opt in node.output:
                    _, tensor = util.get_value_info(opt, submodel)
                    vluList.append(tensor)
                pe_node.append(node.name)
            
        PEDict[f'Node_{nodeList[i]}'] = pe_node
    
    graph_def = onnx.helper.make_graph(
            nodes=nodList,
            name=f"version transform",
            inputs=iptList,   # Graph input
            outputs=optList,  # Graph output
            value_info=vluList,
        ) 

    # Create Model
    model_def = onnx.helper.make_model(graph_def, producer_name="acai-lab16")
    model_def.opset_import[0].version = 7
    onnx.save(model_def,f'out/googlenet-v12_no_dropout/PE{pivot}.onnx')

    with open(f'out/googlenet-v12_no_dropout/PE{pivot}.json', 'w') as outfile:
        json.dump(PEDict, outfile, indent=4)   


#
PE0 = {}
PE1 = {}
PE2 = {}
PE3 = {}

with open(f'out/googlenet-v12_no_dropout/PE0.json', 'r') as outfile:
    PE0 = json.load(outfile)
with open(f'out/googlenet-v12_no_dropout/PE1.json', 'r') as outfile:
    PE1 = json.load(outfile)
with open(f'out/googlenet-v12_no_dropout/PE2.json', 'r') as outfile:
    PE2 = json.load(outfile)
with open(f'out/googlenet-v12_no_dropout/PE3.json', 'r') as outfile:
    PE3 = json.load(outfile)


print("| ", "-".center(72, "-")," |")
print("|",f"PE0".center(17), "|",end='')
print(f"PE1".center(17), "|",end='')
print(f"PE2".center(17), "|",end='')
print(f"PE3".center(17), "|")
for node0, node1, node2, node3 in zip(PE0,PE1,PE2,PE3):
   for sub0, sub1, sub2, sub3 in zip(PE0[node0],PE1[node1],PE2[node2],PE3[node3]):
        if sub0 == sub1 and sub1 == sub2 and sub2 == sub3:
            print("|",f"{sub0}".center(74), "|")
        else:  
            print("| ", "==".center(72, "=")," |")
            
            print("|",f"{sub0}".center(17), "|",end='')
            print(f"{sub1}".center(17), "|",end='')
            print(f"{sub2}".center(17), "|",end='')
            print(f"{sub3}".center(17), "|")
                
            
print("| ", "-".center(72, "-")," |")
            
            
            
# python3 main.py -l NCHW -m model/format-v7/googlenet-v12_no_dropout.onnx -f infershape
# python3 googlenet_4PE_subgraph.py
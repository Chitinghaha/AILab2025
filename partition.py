import onnx
from src.mapping.fusion_partition.CONV_RELU_MAXPOOL_Fusion_and_Partition import CONV_RELU_MAXPOOL_Fusion_and_Partition
from src.mapping.fusion_partition.CONV_RELU_Fusion_and_Partition import  CONV_RELU_Fusion_and_Partition
from src.mapping.fusion_partition.RELU_GAP_FLATTEN_GEMM_Fusion_and_Partition import RELU_GAP_FLATTEN_GEMM_Fusion_and_Partition
import os 

model = onnx.load(os.path.join(os.getcwd(),'model/format-v7/resnet50-v14.onnx'))
partition = 4
model = CONV_RELU_MAXPOOL_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/conv1/Conv",relu_node_name="/relu/Relu",mxpo_node_name="/maxpool/MaxPool",concat_node_name="/concat/concat") 
print("1.0")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.0/conv1/Conv",relu_node_name="/layer1/layer1.0/relu/Relu",concat_node_name="/layer1/layer1.0/concat/Concat1") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.0/conv2/Conv",relu_node_name="/layer1/layer1.0/relu_1/Relu",concat_node_name="/layer1/layer1.0/concat_1/Concat1") 
print("1.1")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.1/conv1/Conv",relu_node_name="/layer1/layer1.1/relu/Relu",concat_node_name="/layer1/layer1.1/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.1/conv2/Conv",relu_node_name="/layer1/layer1.1/relu_1/Relu",concat_node_name="/layer1/layer1.1/concat_1/Concat") 
print("1.2")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.2/conv1/Conv",relu_node_name="/layer1/layer1.2/relu/Relu",concat_node_name="/layer1/layer1.2/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer1/layer1.2/conv2/Conv",relu_node_name="/layer1/layer1.2/relu_1/Relu",concat_node_name="/layer1/layer1.2/concat_1/Concat") 
print("2.0")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.0/conv1/Conv",relu_node_name="/layer2/layer2.0/relu/Relu",concat_node_name="/layer2/layer2.0/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.0/conv2/Conv",relu_node_name="/layer2/layer2.0/relu_1/Relu",concat_node_name="/layer2/layer2.0/concat_1/Concat") 
print("2.1")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.1/conv1/Conv",relu_node_name="/layer2/layer2.1/relu/Relu",concat_node_name="/layer2/layer2.1/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.1/conv2/Conv",relu_node_name="/layer2/layer2.1/relu_1/Relu",concat_node_name="/layer2/layer2.1/concat_1/Concat")  
print("2.2")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.2/conv1/Conv",relu_node_name="/layer2/layer2.2/relu/Relu",concat_node_name="/layer2/layer2.2/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.2/conv2/Conv",relu_node_name="/layer2/layer2.2/relu_1/Relu",concat_node_name="/layer2/layer2.2/concat_1/Concat") 
print("2.3")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.3/conv1/Conv",relu_node_name="/layer2/layer2.3/relu/Relu",concat_node_name="/layer2/layer2.3/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer2/layer2.3/conv2/Conv",relu_node_name="/layer2/layer2.3/relu_1/Relu",concat_node_name="/layer2/layer2.3/concat_1/Concat") 
print("3.0")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.0/conv1/Conv",relu_node_name="/layer3/layer3.0/relu/Relu",concat_node_name="/layer3/layer3.0/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.0/conv2/Conv",relu_node_name="/layer3/layer3.0/relu_1/Relu",concat_node_name="/layer3/layer3.0/concat_1/Concat") 
print("3.1")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.1/conv1/Conv",relu_node_name="/layer3/layer3.1/relu/Relu",concat_node_name="/layer3/layer3.1/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.1/conv2/Conv",relu_node_name="/layer3/layer3.1/relu_1/Relu",concat_node_name="/layer3/layer3.1/concat_1/Concat")
print("3.2")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.2/conv1/Conv",relu_node_name="/layer3/layer3.2/relu/Relu",concat_node_name="/layer3/layer3.2/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.2/conv2/Conv",relu_node_name="/layer3/layer3.2/relu_1/Relu",concat_node_name="/layer3/layer3.2/concat_1/Concat")
print("3.3")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.3/conv1/Conv",relu_node_name="/layer3/layer3.3/relu/Relu",concat_node_name="/layer3/layer3.3/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.3/conv2/Conv",relu_node_name="/layer3/layer3.3/relu_1/Relu",concat_node_name="/layer3/layer3.3/concat_1/Concat")
print("3.4")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.4/conv1/Conv",relu_node_name="/layer3/layer3.4/relu/Relu",concat_node_name="/layer3/layer3.4/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.4/conv2/Conv",relu_node_name="/layer3/layer3.4/relu_1/Relu",concat_node_name="/layer3/layer3.4/concat_1/Concat")
print("3.5")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.5/conv1/Conv",relu_node_name="/layer3/layer3.5/relu/Relu",concat_node_name="/layer3/layer3.5/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer3/layer3.5/conv2/Conv",relu_node_name="/layer3/layer3.5/relu_1/Relu",concat_node_name="/layer3/layer3.5/concat_1/Concat")
print("4.0")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.0/conv1/Conv",relu_node_name="/layer4/layer4.0/relu/Relu",concat_node_name="/layer4/layer4.0/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.0/conv2/Conv",relu_node_name="/layer4/layer4.0/relu_1/Relu",concat_node_name="/layer4/layer4.0/concat_1/Concat")
print("4.1")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.1/conv1/Conv",relu_node_name="/layer4/layer4.1/relu/Relu",concat_node_name="/layer4/layer4.1/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.1/conv2/Conv",relu_node_name="/layer4/layer4.1/relu_1/Relu",concat_node_name="/layer4/layer4.1/concat_1/Concat")
print("4.2")
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.2/conv1/Conv",relu_node_name="/layer4/layer4.2/relu/Relu",concat_node_name="/layer4/layer4.2/concat/Concat") 
model = CONV_RELU_Fusion_and_Partition(partition=partition,model=model,conv_node_name="/layer4/layer4.2/conv2/Conv",relu_node_name="/layer4/layer4.2/relu_1/Relu",concat_node_name="/layer4/layer4.2/concat_1/Concat")


model = RELU_GAP_FLATTEN_GEMM_Fusion_and_Partition(
    partition=partition,
    model=model,
    relu_node_name = "/layer4/layer4.2/relu_2/Relu",
    gapl_node_name = "/avgpool/GlobalAveragePool",
    fltn_node_name = "/Flatten",
    gemm_node_name = "/fc/Gemm",
    splt_node_name = "/layer4/layer4.2/split/Split",
    sums_node_name = "/fc/Sum")



onnx.save(model,os.path.join(os.getcwd(),"out","format-v7",f"resnet50-v14_{partition}_core.onnx"))
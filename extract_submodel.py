
import sys
import argparse
parser = argparse.ArgumentParser()
from src.tool import util
import os
import onnx
parser.add_argument('-m','--modelpath', required=True, help='onnx model path')
parser.add_argument('-o','--subgraph', required=True, help='onnx submodel model path')
parser.add_argument('-f','--first', required=True, help='onnx submodel model path')
parser.add_argument('-e','--end', required=True, help='onnx submodel model path')
args = parser.parse_args()

model = onnx.load(args.modelpath)
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
submodel = util.extract_subgraph_node2node(model,firstNode=args.first,finalNode=args.end)
dirPath, _ = os.path.splitext(args.subgraph)
modelname = os.path.basename(dirPath)

    
if not os.path.exists(os.path.join(os.getcwd(), "out", modelname)):
    os.makedirs(os.path.join(os.getcwd(), "out", modelname))
if os.path.isfile(os.path.join(os.getcwd(), "out", modelname, args.subgraph)):
   os.remove(os.path.join(os.getcwd(), "out", modelname, args.subgraph)) 
onnx.save(submodel, os.path.join(os.getcwd(), "out", modelname, args.subgraph))
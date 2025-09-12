import onnx
from .op import Conv, Relu, template, MaxPool, GlobalAveragePool,Flatten,Gemm ,Add
from typing import Optional, Union
def analyticalModel(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:list = None, csvPath:str = None) -> tuple([int, dict,list]):
    operator = {
        "Conv" : Conv.analysis,
        "Relu" : Relu.analysis,
        "MaxPool": MaxPool.analysis,
        "Add" :  Add.analysis,
        "GlobalAveragePool" : GlobalAveragePool.analysis,
        "Flatten" : Flatten.analysis,
        "Gemm" : Gemm.analysis,
        "Concat" : template.analysis,
        "Sum" : template.analysis,
        }
    if node.op_type not in operator.keys():
        raise BaseException(f"Analytical Model : \'{node.op_type}\' doesn't exist." )
    else:
        return operator[node.op_type](model, layout, node, memoryTable, csvPath)

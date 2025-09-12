from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict, list]):
    memory = 0
    cycle = 0
    memoryRequest = 0
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable
from . import *
from .anal import anal
from typing import Optional
from src.cost.anal import anal
from src.management import tool
from src.structure import DATA_SIZE_DTYPE
def costFunction(model:onnx.ModelProto,layout:str,  memoryTable:Optional[list] = None, csvPath:Optional[str] = None):
    
    
    memoryRequest = 0
    for tensor in model.graph.input:
        dims = [memory.dim_value for memory in tensor.type.tensor_type.shape.dim]
        memory = DATA_SIZE_DTYPE[tensor.type.tensor_type.elem_type]
        for dim in dims:
            memory *= dim
        request, memoryTable =  tool.malloc(tensor.name, memory // 8, memoryTable)
        memoryRequest += request
    # for tensor in model.graph.initializer:
    #     dims = tensor.dims
    #     memory = DATA_SIZE_DTYPE[tensor.data_type]
    #     for dim in dims:
    #         memory *= dim
    #     request, memoryTable =  tool.malloc(tensor.name, memory // 8, memoryTable)
    #     memoryRequest += request 


    cycle = 0
    _ = tool.dump_csv(csvPath, memoryTable, config.MEMORY_SIZE, 0)
    nodes = [node for node in model.graph.node]
    for node in nodes:
        _, table ,memoryTable = anal.analyticalModel(
            model=model,
            layout=layout,
            node=node,
            memoryTable=memoryTable,
            csvPath=csvPath
            )
        cycle += table['cycle']
        # memoryRequest += request
    _ = tool.dump_csv(csvPath, memoryTable, config.MEMORY_SIZE, 0)
    return cycle
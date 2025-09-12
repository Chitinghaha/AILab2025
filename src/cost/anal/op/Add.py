from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    memoryRequest = 0
    _, A = get_value_info(node.input[0], model)
    
    _, B = get_value_info(node.input[1], model)
    _, C = get_value_info(node.output[0], model)
    
    
    dimA = [a.dim_value for a in A.type.tensor_type.shape.dim]
    dimB = [a.dim_value for a in B.type.tensor_type.shape.dim]
    dimC = [a.dim_value for a in C.type.tensor_type.shape.dim]
    if layout == "NHWC":
        dimA[1], dimA[3] = dimA[3], dimA[1]
        dimB[1], dimB[3] = dimB[3], dimB[1]
        dimC[1], dimC[3] = dimC[3], dimC[1] 
        
    typeA = A.type.tensor_type.elem_type    
    typeB = B.type.tensor_type.elem_type
    typeC = C.type.tensor_type.elem_type
    
    staticMemA = DATA_SIZE_DTYPE[typeA]
    staticMemB = DATA_SIZE_DTYPE[typeB]
    staticMemC = DATA_SIZE_DTYPE[typeC]
    
    
    for dim in dimA: staticMemA *= dim
    for dim in dimB: staticMemB *= dim
    for dim in dimC: staticMemC *= dim

    loop = dimA[0] * dimA[1] * dimA[2] * dimA[3]
    
    # SW instruction :
    store = dimC[0] * dimC[1] * dimC[2] * dimC[3]
    # LW instruction :
    load = dimA[0] * dimA[1] * dimA[2] * dimA[3]

    create_weight_address =  2 + 3 # 3 (multiply) + 3 (addition)
    create_input_address =  2 + 3 
    create_output_address = 2 + 3 # 2 (multiply) + 3 (addition)
    create_temp = 2 + 3 # 2 (multiply) + 3 (addition)
    
    # create Input
    create_input = (config.DATA_LATENCY + create_input_address) * load
    
    # Create weight
    create_weight = (config.DATA_LATENCY + create_weight_address) * load
    
    # Create Output
    create_output = (config.DATA_LATENCY + create_output_address) * store + create_temp * loop
    
    # Number of branch
    branch_count = 0
    # CPU instruction count :
    cycle = create_input + create_weight + create_output + branch_count
    # Memory Requirement
    memory = staticMemA + staticMemB + staticMemC
    

    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemC // 8 , memoryTable)
    memoryRequest += request
    tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE_IN_LAB16_3+1, second=cycle)
    # for ipt in node.input:
    #     memoryTable = tool.free(ipt, memoryTable)
    memoryTable = tool.free(node.input[0], memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable
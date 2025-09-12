from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    memoryRequest = 0
    _, X = get_value_info(node.input[0], model)
    
    _, W = get_initilizer(node.input[1], model)
    _, Y = get_value_info(node.output[0], model)
    
    if len(node.input) == 3:
        _, B = get_initilizer(node.input[2], model)
    
    dimX = [a.dim_value for a in X.type.tensor_type.shape.dim]
    dimY = [a.dim_value for a in Y.type.tensor_type.shape.dim]
    dimW = W.dims
    
    # if layout == "NHWC":
    #     dimX[1], dimX[3] = dimX[3], dimX[1]
    #     dimW[1], dimW[3] = dimW[3], dimW[1]
    #     dimY[1], dimY[3] = dimY[3], dimY[1] 
    
    typeX = X.type.tensor_type.elem_type    
    typeY = Y.type.tensor_type.elem_type
    typeW = W.data_type
    
    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    staticMemW = DATA_SIZE_DTYPE[typeW]
    
    
    for dim in dimX: staticMemX *= dim
    for dim in dimY: staticMemY *= dim
    for dim in dimW: staticMemW *= dim
    
    # print(dimX)
    # print(dimY)
    # print(dimW)

    loop = dimX[0] * dimX[1] * dimY[1] 
    
    # SW instruction :
    store = dimY[0] * dimY[1]
    # LW instruction :
    load = dimX[0] * dimX[1] 

    create_weight_address =  2 + 2 # 2 (multiply) + 2 (addition)
    create_input_address =  2 + 2 # 2 (multiply) + 2 (addition)
    create_output_address = 2 + 2 # 2 (multiply) + 2 (addition)
    create_temp = 2 + 2 # 2 (multiply) + 2 (addition)

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
    memory = staticMemX + staticMemY + staticMemW
    

    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE_IN_LAB16_3+1, second=cycle)
    # for ipt in node.input:
    #     memoryTable = tool.free(ipt, memoryTable)
    memoryTable = tool.free(node.input[0], memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable
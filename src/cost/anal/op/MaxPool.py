from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    memory = 0
    cycle = 0
    memoryRequest = 0
    
    _, X = get_value_info(node.input[0], model)
    _, Y = get_value_info(node.output[0], model)
    dimX = [a.dim_value for a in X.type.tensor_type.shape.dim]
    dimY = [a.dim_value for a in Y.type.tensor_type.shape.dim]
    if layout == "NHWC":
        dimX[1], dimX[3] = dimX[3], dimX[1]
        dimY[1], dimY[3] = dimY[3], dimY[1] 
    typeX = X.type.tensor_type.elem_type    
    typeY = Y.type.tensor_type.elem_type
    
    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    
    for dim in dimX: staticMemX *= dim
    for dim in dimY: staticMemY *= dim
    
    
    attr_dict = get_attribute(node.attribute)

    # ceil_mode = attr_dict.get("ceil_mode")
    kernel_shape = attr_dict.get("kernel_shape")
    # pads = attr_dict.get("pads")
    # strides = attr_dict.get("strides")

    
    # print(kernel_shape)

    
    # SW instruction :
    store = dimY[0] * dimY[1] * dimY[2] * dimY[3]
    # LW instruction :
    load = dimX[0] * dimX[1] * dimX[2] * dimX[3]
    
    
    create_input_address =  2 + 3
    create_output_address = 2 + 3
    
    # create Input
    create_input = (config.DATA_LATENCY + create_input_address) * load
    
    # maxpool()
    create_temp = (kernel_shape[0]*kernel_shape[1] - 1) * store  #if compaere is one cycles
    
    # Create Output
    create_output = (config.DATA_LATENCY + create_output_address) * store
    
    # CPU instruction count :
    cycle = create_input + create_temp + create_output 
    
    # Memory Requirement
    memory = staticMemX + staticMemY

    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE_IN_LAB16_3+1, second=cycle)
    # for ipt in node.input:
    #     memoryTable = tool.free(ipt, memoryTable)
    memoryTable = tool.free(node.input[0], memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable
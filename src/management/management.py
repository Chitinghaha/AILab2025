
import onnx 
from src.scheduler.scheduler import scheduler
from src.tool.util import create_operator_list_dict, create_tensor_dict
from src.management import tool
from src.structure import DATA_SIZE_DTYPE
import os
import src.config as config

def opt_manager(model:onnx.ModelProto, operatorPath:str, csvPath:str) -> int:
    
    activative_tensor, static_tensor = create_tensor_dict(model)
    nodeList, nodeDict = create_operator_list_dict(model, static_tensor)
    topo_order = scheduler(nodeDict, nodeList)
    
    # 找出每個 tensor 第一次被產生(作為 output)與最後一次被使用(作為 input)的 operator index
    tensor_live_range = {}
    
    # 紀錄 tensor 首次產生與最後使用的 step
    for step, node_idx in enumerate(topo_order):
        op_name = nodeList[node_idx]
        inputs = nodeDict[op_name]['input']
        outputs = nodeDict[op_name]['output']
        
        # tensor 第一次產生時間 (output)
        for t in outputs:
            if t not in tensor_live_range:
                tensor_live_range[t] = {'start': step, 'end': step}
            else:
                # 防止初始化就出現
                if step < tensor_live_range[t]['start']:
                    tensor_live_range[t]['start'] = step
        
        # tensor 最後使用時間 (input)
        for t in inputs:
            if t not in tensor_live_range:
                tensor_live_range[t] = {'start': step, 'end': step}
            else:
                if step > tensor_live_range[t]['end']:
                    tensor_live_range[t]['end'] = step
    
    # 對模型輸入 tensor，視為從 step -1 開始活躍
    model_inputs = [t.name for t in model.graph.input if t.name not in [x.name for x in model.graph.initializer]]
    for t in model_inputs:
        if t not in tensor_live_range:
            tensor_live_range[t] = {'start': -1, 'end': -1}
        else:
            tensor_live_range[t]['start'] = -1
    
    # 計算每個時間點活躍 tensor 總大小
    max_mem = 0
    # time 從 -1 到最大 step
    max_step = max(topo_order) if len(topo_order) > 0 else 0
    for time in range(-1, len(topo_order)):
        current_mem = 0
        for tensor, live in tensor_live_range.items():
            if live['start'] <= time <= live['end']:
                size = tool.operator_Mem_Bytes(activative_tensor[tensor])
                current_mem += size
        if current_mem > max_mem:
            max_mem = current_mem
            
    mem_kb = max_mem / 1024 
    
    print(f"Optimal memory lower bound: {mem_kb} KB")
    
    return max_mem



def manager(model:onnx.ModelProto, operatorPath:str, csvPath:str) -> int:
    activative_tensor, static_tensor = create_tensor_dict(model)
    nodeList, nodeDict = create_operator_list_dict(model, static_tensor)
    ##################################
    #         Topoligial Sort        #
    topo_order =  scheduler(nodeDict, nodeList)
    ##################################
    memoryTable = [{"valid":0, "address":0, "size":config.MEMORY_SIZE, "tensor":""}]
    init = [ tensor.name for tensor in model.graph.initializer]
    modelinputList = []
    for tensor in model.graph.input:
        if tensor.name not in init:
            modelinputList.append(tensor.name)
    del init
    
    
    opt_max = opt_manager(model, operatorPath, csvPath)
    
    memMAX = 0
    for inputName in modelinputList:
        inputSize = tool.operator_Mem_Bytes(activative_tensor[inputName])
        _, memoryTable = tool.malloc(inputName,inputSize,memoryTable,opt_max)
    
    
    if os.path.isfile(csvPath):
        os.remove(csvPath)
    second = 0 
    memMAX = tool.dump_csv(csvPath, memoryTable,memMAX, second)
    
            
    for operator in topo_order:
        operatorName = nodeList[operator]
        inputList = nodeDict[operatorName]['input']
        initsList = nodeDict[operatorName]['initializer']
        outputList = nodeDict[operatorName]['output']
        # fullPath = os.path.join(operatorPath,operatorName+".onnx")    
        # second = reTimeSingleOperator(fullPath)
        second += 1
        for tensorName in outputList:
            memory = tool.operator_Mem_Bytes(activative_tensor[tensorName])
            _, memoryTable = tool.malloc(tensorName, memory, memoryTable,opt_max)
        memMAX = tool.dump_csv(csvPath,memoryTable, memMAX, second)
        for tensorName in inputList:
            activative_tensor[tensorName]['consumer'].remove(operatorName)
            if len(activative_tensor[tensorName]['consumer']) == 0:
                memoryTable = tool.free(tensorName, memoryTable) 

    return memMAX
            
    

    

    

    










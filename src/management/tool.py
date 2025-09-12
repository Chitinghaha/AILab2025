
from src.structure import DATA_SIZE_DTYPE
from src.config import MEMORY_SIZE
from . import policy 
import csv
import os
# memoryTable = [{"valid":0, "address":0, "size":4194304, "tensor":""}]

# def malloc(tensorName:str, tensorSize:int, memoryTable:list)->tuple([int,list]):
#     pivot = policy.hit(memoryTable, tensorSize, MEMORY_SIZE + 1, hit_policy="excellentHit") # 32MB + 1
#     if pivot == -1:
#         print(f"Out of Memory : {tensorName} : {tensorSize/1024} KB")
#         if memoryTable[-1]['valid'] == 1:
#             return tensorSize, memoryTable
#         else:  
#             return tensorSize - memoryTable[-1]["size"], memoryTable
#     else:
#         oBlk = memoryTable[pivot]
#         nBlk = {"valid":1, "address":oBlk['address'], "size":tensorSize, "tensor": tensorName}
#         memoryTable[pivot]["address"] = oBlk['address'] + tensorSize
#         memoryTable[pivot]["size"]    = oBlk['size'] - tensorSize
#         memoryTable.insert(pivot,nBlk)
#         return 0, memoryTable

def malloc(tensorName: str, tensorSize: int, memoryTable: list,opt_max:int) -> tuple([int, list]):
    pivot = policy.hit(memoryTable, tensorSize, MEMORY_SIZE + 1, hit_policy="FirstHit")  # 32MB + 1
    # print(f"tensorName: {tensorName}, pivot: {pivot}, tensorSize: {tensorSize}")
    if(len(memoryTable) == 1 ):
    
            # print("len(memoryTable) ")
            nBlk = {
                "valid": 0,
                "address": opt_max ,
                "size": memoryTable[0]['size'] - opt_max,
                "tensor": memoryTable[0]['tensor']
            }
            
            memoryTable[0]['size'] = opt_max
            
            memoryTable.insert(pivot + 1, nBlk)
            # print(memoryTable)
    

    if pivot == -1:
        # print(f"Out of Memory : {tensorName} : {tensorSize / 1024} KB")
        if memoryTable[-1]['valid'] == 1:
            return tensorSize, memoryTable
        else:
            return tensorSize - memoryTable[-1]["size"], memoryTable
    
    else:
        oBlk = memoryTable[pivot]
        threshold = oBlk['size'] * 0.2  
            
        
        if tensorSize < threshold: 
            nBlk = {
                "valid": 1,
                "address": oBlk['address'],
                "size": tensorSize,
                "tensor": tensorName
            }
            oBlk['address'] += tensorSize
            oBlk['size'] -= tensorSize
            memoryTable.insert(pivot, nBlk)
        else:
            
            nBlk = {
                "valid": 1,
                "address": oBlk['address'] + oBlk['size'] - tensorSize,
                "size": tensorSize,
                "tensor": tensorName
            }
            oBlk['size'] -= tensorSize
            memoryTable.insert(pivot + 1, nBlk)
        
        # print(memoryTable)

        return 0, memoryTable


def free(tensorName:str,  memoryTable:list)->list:
    for cnt in range(len(memoryTable)):
        if memoryTable[cnt]['tensor'] == tensorName:
            block =  memoryTable[cnt]
            if cnt > 0 and cnt < (len(memoryTable) - 1): # 1 ~ (n-1)
                if memoryTable[cnt + 1]['valid'] == 0 and memoryTable[cnt - 1]['valid'] == 0:  # merge [cnt-1 cnt cnt+1] block
                    memoryTable[cnt - 1]['size'] += (block['size'] + memoryTable[cnt + 1]['size'])
                    memoryTable.remove(memoryTable[cnt + 1])
                    memoryTable.remove(block)
                elif memoryTable[cnt + 1]['valid'] != 0 and memoryTable[cnt - 1]['valid'] == 0: # merge [cnt-1 cnt] block
                    memoryTable[cnt - 1]['size'] += block['size']
                    memoryTable.remove(block)
                elif memoryTable[cnt + 1]['valid'] == 0 and memoryTable[cnt - 1]['valid'] != 0: # merge [cnt-1 cnt] block
                    memoryTable[cnt + 1]['address'] = block['address']
                    memoryTable[cnt + 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            elif cnt == 0 and cnt < len(memoryTable): # 0
                if memoryTable[cnt + 1]['valid'] == 0:
                    memoryTable[cnt + 1]['address'] = block['address']
                    memoryTable[cnt + 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            elif cnt > 0 and cnt == (len(memoryTable) - 1): # -1
                if memoryTable[cnt - 1]['valid'] == 0:
                    memoryTable[cnt - 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            break
    return memoryTable


def dump_csv(csvPath:str,memoryTable:dict,memMAX:int,second:int) -> int:
    
    with open(csvPath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [second]
            for m in memoryTable:
                if m['valid'] == 0 and m['tensor'] == '':
                    row.append('empty')
                    row.append(m['address'])
                    row.append(m['address'] + m['size'])
                else:
                    row.append(m['tensor'])
                    row.append(m['address']) 
                    row.append(m['address'] + m['size'])
                    if m['address'] + m['size'] > memMAX: memMAX = m['address'] + m['size']
            writer.writerow(row)
    return memMAX

def operator_Mem_Bytes(tensorInfo):
    memory = 1
    for dim in tensorInfo['dims']: memory *= dim
    return memory * DATA_SIZE_DTYPE[tensorInfo['element_type']] // 8 




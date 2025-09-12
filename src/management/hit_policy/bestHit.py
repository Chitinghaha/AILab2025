def hit(memoryTable:list,memory:int,memory_MAX:int)->int:
    hit_index = -1
    bestRecord = memory_MAX
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if (block['valid'] == 1):
            continue
        else:
            if block['size'] >= memory and block['size'] < bestRecord:
                hit_index = cnt
                bestRecord = block['size']
    return hit_index

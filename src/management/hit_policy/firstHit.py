def hit(memoryTable:list,memory:int,memory_MAX:int)->int:
    hit_index = -1
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if (block['valid'] == 1):
            continue
        else:
            if block['size'] >= memory:
                hit_index = cnt
                break
    return hit_index 
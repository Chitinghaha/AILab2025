def hit(memoryTable:list,memory:int,memory_MAX:int)->int:
    hit_index = -1
    bestRecord = memory_MAX
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if (block['valid'] == 1):
            continue
        else:
            if block['size'] >= memory and block['size'] < bestRecord :
                hit_index = cnt
                bestRecord = block['size']
    return hit_index




# python3 main.py -l NCHW -m out/googlenet-v12_no_dropout/googlenet-v12_no_dropout.onnx -f management
# python3 gen_graph.py -c out/googlenet-v12_no_dropout/memory.csv -o out/googlenet-v12_no_dropout/memory_allocation.png
# python3 main.py -l NCHW -m model/format-v7/googlenet-v12_no_dropout.onnx -f infershape 


# python3 main.py -l NCHW -m out/resnet50-v14/resnet50-v14.onnx -f management
# python3 gen_graph.py -c out/resnet50-v14/memory.csv -o out/resnet50-v14/memory_allocation.png
#!/bin/bash

for ((i=0;i < 50;++i)) do
    echo "resnet50_spatial_partition.py out/resnet50-v14/subgraph/temporal/$i.onnx"
    python3 resnet50_spatial_partition.py -m out/resnet50-v14/subgraph/temporal/$i.onnx -o out/resnet50-v14/subgraph/ -c 4
done



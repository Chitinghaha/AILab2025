#!/bin/bash

for ((i=0;i < 82;++i)) do
    echo "googlenet_spatial_partition.py out/googlenet-v12_no_dropout/subgraph/temporal/$i.onnx"
    python3 googlenet_spatial_partition.py -m out/googlenet-v12_no_dropout/subgraph/temporal/$i.onnx -o out/googlenet-v12_no_dropout/subgraph/ -c 4
done



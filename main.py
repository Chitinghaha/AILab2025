
import sys
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-m','--modelpath', required=True, help='onnx model path')
parser.add_argument('-f','--function', required=True, help='choose the function : slice / infershape / empirical / analytical / management / cost / upgrade')
parser.add_argument('-l','--layout', required=True, help='select the layout: NCHW / NHWC')
parser.add_argument('-o','--csv', default=None,required=False, help='csv file')
parser.add_argument('-p','--partition', default=None,required=False, help='PE Number')
args = parser.parse_args()

import src.funtion as f

layoutList = ["NCHW", "NHWC"]
functionDict = {
        "upgrade": f.upgrade,
        "slice" : f.slice,
        "infershape" : f.infershape,
        "empirical" : f.empirical,
        "analytical" :f.analytical, 
        'management':f.management,
        "cost": f.cost,
    }

if (
    (args.function not in functionDict.keys())
    or (args.layout not in layoutList)
):
    print("Invalid usage.", file=sys.stderr)
    parser.print_help()
    sys.exit(-1)
else:
    if not os.path.isdir("out"):
        os.mkdir("out/")
    functionDict[args.function](args, parser)



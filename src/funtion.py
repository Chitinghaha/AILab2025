import os
import onnx
import src.tool.util as util
import src.config as config
from src.cost.anal import anal
from  src.cost.costFunction import costFunction 
import src.management.management as mngt
import sys
def slice(args, parser): # Checked
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    submodelDirPath = os.path.join(os.getcwd(), "out", modelname, "operator")
    if not os.path.exists(submodelDirPath):
        os.makedirs(os.path.join(os.getcwd(), "out", modelname , "operator"))
    util.Slice_Node(args.modelpath, submodelDirPath)

def infershape(args, parser): # Checked
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    submodelDirPath = os.path.join(os.getcwd(), "out")
    batch = input("please enter the batch Size : ")
    c, h, w = input("please enter the input dim (c,h,w) : ").split()
    model = util.model_infer_shape(args.modelpath,batch=int(batch),input_shape=[int(c),int(h),int(w)])
    if not os.path.exists(os.path.join(submodelDirPath , modelname)):
        os.makedirs(os.path.join(submodelDirPath , modelname))
    onnx.save(model, os.path.join(submodelDirPath , modelname, modelname + ".onnx"))

def empirical(args, parser): # Checked
    model = onnx.load(args.modelpath)
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    memory = util.reMemsSingleOperator(args.modelpath)
    second = util.reTimeSingleOperator(args.modelpath)
    print(f"\n=> Empirical Measurement ")
    print(f" --- {modelname}")
    print(f"  | Memory : {memory:6.2f} KiB")
    print(f"  | Times  : {second:6.2f} ms")

def analytical(args, parser): # checked
    model = onnx.load(args.modelpath)
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    model = onnx.load(args.modelpath)
    csvPath = "out/"+modelname+"/memory.csv"   

    if not os.path.isdir("out/"+modelname):
        os.makedirs("out/"+modelname)
    memoryTable = [{"valid":0, "address":0, "size":config.MEMORY_SIZE+1, "tensor":""}]
    _, info, memoryTable = anal.analyticalModel(model=model, layout=args.layout, node=model.graph.node[0],memoryTable=memoryTable,csvPath=csvPath)
    name = model.graph.node[0].name
    print(f"\n=> Analytical Measurement in Scalar CPU")
    print(f" --- {name}")
    print(f"  | Memory : {info['memory']:6.2f} KiB")
    print(f"  | Times  : {info['cycle']:6.0f} cycles")

def management(args, parser): # checked
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    model = onnx.load(args.modelpath)
    operatorPath = "out/"+modelname+"/operator"
    csvPath = "out/"+modelname+"/memory.csv"
    if not os.path.isdir("out/"+modelname):
        os.makedirs("out/"+modelname)
    memMAX = mngt.manager(model, operatorPath,csvPath)
    print(f"Memory Requirement : {memMAX / (1024)} KB")

def cost(args, parser):
    model = onnx.load(args.modelpath)
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    if not os.path.isdir("out/"+modelname):
        os.makedirs("out/"+modelname)
    memoryTable = [{"valid":0, "address":0, "size":config.MEMORY_SIZE_IN_LAB16_3, "tensor":""}]
    csvPath = "out/"+modelname+"/memory.csv"
    if os.path.isfile(csvPath):
        os.remove(csvPath)
    cycle = costFunction(model, layout= args.layout, memoryTable=memoryTable,csvPath=csvPath)
    print(f"cycle count : {cycle}")


    

def mapping(args, parser):
    if args.partition == None:
        print("PE Number is missing\n")
        parser.print_help()
        sys.exit(-1)
    pass

def upgrade(args, parser):
    
    model = onnx.load(args.modelpath)
    dirPath, _ = os.path.splitext(args.modelpath)
    modelname = os.path.basename(dirPath)
    submodelDirPath = os.path.join(os.getcwd(), "out", modelname)    
    if not os.path.isdir(submodelDirPath):
        os.makedirs(submodelDirPath)
    iptList = []
    optList = []
    intList = []
    vluList = []
    intNameList = []
    nodeList = []
    for init in model.graph.initializer:
        intNameList.append(init.name)
        intList.append(init)
    for ipt in model.graph.input:
        if ipt not in intNameList:
            iptList.append(ipt)
    for opt in model.graph.output:
        optList.append(opt) 
    for value in model.graph.value_info:
        vluList.append(value) 
    for idx, node in enumerate(model.graph.node):
        node.name = node.op_type +"_"+ str(idx)
        nodeList.append(node)
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodeList,
        name=f"version transform",
        inputs=iptList,   # Graph input
        outputs=optList,  # Graph output
        initializer=intList,
        value_info=vluList,
    ) 
    # Create Model
    model_def = onnx.helper.make_model(graph_def, producer_name="acai-lab16")
    model_def.opset_import[0].version = 7
    inferred_model = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, os.path.join(submodelDirPath, modelname + "_v7.onnx"))
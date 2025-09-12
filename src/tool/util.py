import onnx
import numpy as np
import onnxruntime
import tracemalloc
from onnx.utils import Extractor
from typing import Union
import time
from onnxruntime.tools import onnx_model_utils
from src.structure import  ONNX_DTYPE, NUMPY_DTYPE
from typing import List

def Slice_Node(modelPath:str, submodelPath:str) -> None:
    """Slice Onnx Model by operator
    
    Args:
        modelPath: onnx model path
        submodelPath: ouput model path
    """
    model = onnx.load(modelPath)
    print(modelPath,submodelPath)
    for i in model.graph.node:
        print(i.name)
        onnx.utils.extract_model(modelPath, submodelPath + f'/{i.name}.onnx', i.input, [i.output[0]])


def model_infer_shape(modelPath:str,batch:int,input_shape:list) -> onnx.ModelProto:
    model = onnx.load(modelPath)
    onnx.checker.check_model(model)
    onnx_model_utils.make_input_shape_fixed(model.graph, model.graph.input[0].name, [batch,input_shape[0],input_shape[1],input_shape[2]])
    inferred_model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)
    return inferred_model

def prepareInput(subgraphPath:str) -> dict:
    model = onnx.load(subgraphPath)
    weight = [ ip.name for ip in model.graph.initializer]
    inputs = {}
    for ipt in model.graph.input:
        if ipt.name not in weight:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            dim = []
            for d in dims:
                dim.append(d.dim_value)
            inputs[ipt.name] = np.ones(tuple(dim),dtype=NUMPY_DTYPE[model.graph.input[0].type.tensor_type.elem_type])
    return inputs


def reMemsSingleOperator(subgraphPath:str) -> float:
    sess = onnxruntime.InferenceSession(subgraphPath)
    tracemalloc.start()

    _, peak = tracemalloc.get_traced_memory()
    _ = sess.run(None, prepareInput(subgraphPath))
    _, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (peak1-peak)/1024

def reTimeSingleOperator(subgraphPath:str) -> int:
    sess = onnxruntime.InferenceSession(subgraphPath)
    start = time.process_time()

    _ = sess.run(None, prepareInput(subgraphPath))
    end = time.process_time()
    return (end - start) * 1000 


def get_value_info(tensor_name:str, model:onnx.ModelProto) ->tuple([int,onnx.ValueInfoProto]):
    tensor_idx = 0
    tensor = None
    for tensor_info, output in enumerate(model.graph.input):
        if  output.name == tensor_name:
            tensor_idx = -1
            tensor = output
            break
    for tensor_info, output in enumerate(model.graph.output):
        if  output.name == tensor_name:
            tensor_idx = -1
            tensor = output
            break
    for tensor_info, output in enumerate(model.graph.value_info):
        if  output.name == tensor_name:
            tensor_idx = tensor_info
            tensor = output
            break
        
    if tensor == None:
        raise BaseException(f"\'{tensor_name}\' Tensor Not Found")
    
    return tensor_idx, tensor

def get_initilizer(tensor_name:str, model:onnx.ModelProto) ->tuple([int,onnx.TensorProto]):
    tensor_idx = 0
    tensor = None
    for tensor_info, output in enumerate(model.graph.initializer):
        if  output.name == tensor_name:
            tensor_idx = tensor_info
            tensor = output
            break
    if tensor == None:
        raise BaseException(f"\'{tensor_name}\' Tensor Not Found")
    return tensor_idx, tensor


def check_tensor_initilizer(tensor_name:str, model:onnx.ModelProto) -> bool:
    isInit = False
    for _, initializer in enumerate(model.graph.initializer):
        if tensor_name == initializer.name:
            isInit = True
            break
    return isInit

def get_node_Info(node_name:str,model:onnx.ModelProto) -> tuple([int, onnx.NodeProto]):
    NodeInfo = None
    node_idx = -1
    for idx, node in enumerate(model.graph.node):
        if node.name == node_name:
            NodeInfo = node
            node_idx = idx
            break
    if NodeInfo == None:
        raise BaseException(f"\'{node_name}\' Node Not Found")
    return node_idx, NodeInfo

def extract_subgraph_node2node(model:Union[onnx.ModelProto,str],firstNode:Union[onnx.NodeProto,str],finalNode:Union[onnx.NodeProto,str]) -> onnx.ModelProto:
    """Extract the ONNX submodel - Node2Node

    Args:
        model : can be onnx.ModelProto or Onnx ModelPath
        firstNode : can be onnx.NodeProto or Node name in ONNX model
        finalNode : can be onnx.NodeProto or Node name in ONNX model
    Returns:
       onnx.ModelProto  : subgraph

    """

    if isinstance(model, str): model = onnx.load(model)
    if isinstance(firstNode, str): _, firstNode = get_node_Info(firstNode, model)
    if isinstance(finalNode, str): _, finalNode = get_node_Info(finalNode ,model)
    inputs = [] 
    if model.ir_version > 3:
        for ipt in firstNode.input:
            if not check_tensor_initilizer(ipt, model):
                inputs.append(ipt)
    else:
        inputs = firstNode.input
    e = Extractor(model)
    extracted = e.extract_model(inputs, finalNode.output)
    onnx.checker.check_model(extracted)
    return extracted

def extract_subgraph_multi_node2node(model:Union[onnx.ModelProto,str],topNodes:List[Union[onnx.NodeProto,str]],btmNodes:List[Union[onnx.NodeProto,str]]) -> onnx.ModelProto:
    """Extract the ONNX submodel -  multiNode 2 multiNode

    Args:
        model : can be onnx.ModelProto or Onnx ModelPath
        topNodes : can be onnx.NodeProto or Node name in ONNX model
        btmNodes : can be onnx.NodeProto or Node name in ONNX model
    Returns:
       onnx.ModelProto  : subgraph

    """
    if isinstance(model, str): model = onnx.load(model)
    
    for idx in range(len(topNodes)):
        if isinstance(topNodes[idx], str): _, topNodes[idx] = get_node_Info(topNodes[idx], model)
    for idx in range(len(btmNodes)):
        if isinstance(btmNodes[idx], str): _, btmNodes[idx] = get_node_Info(btmNodes[idx], model)    
    inputs = [] 
    
    value_output = []
    
    for node in topNodes:
        for opt in node.output:
            value_output.append(opt)
    
    for node in topNodes:
        if model.ir_version > 3:
            for ipt in node.input:
                if not check_tensor_initilizer(ipt, model) and ipt not in value_output:
                    inputs.append(ipt)
        else:
            for ipt in node.input:
                if ipt not in value_output:
                    inputs.append(ipt)
    outputs = []
    for node in btmNodes:  
        for opt in node.output:
                outputs.append(opt)
    e = Extractor(model)
    extracted = e.extract_model(inputs, outputs)
    onnx.checker.check_model(extracted)
    return extracted


def output_shape(partition_num:int,shape:int)->list:
    number = [0] * partition_num    
    for i in range(shape):
        number[i % partition_num] +=1
    return number

def partition_numpy_kernel(number:list, partition_idx:int, tensor:np.ndarray) -> tuple([list, np.ndarray]):
    shape = tensor.shape
    minn = sum(number[0:partition_idx])
    maxx = sum(number[0:partition_idx+1])
    return [maxx-minn,shape[1],shape[2],shape[3]] , tensor[minn:maxx,:,:,:]

def partition_output(number:list, partition_idx:int, tensor:onnx.ValueInfoProto) -> tuple([int, list]):
    elem_type = tensor.type.tensor_type.elem_type
    shape = [dims.dim_value for dims in tensor.type.tensor_type.shape.dim]
    shape[1] = number[partition_idx]
    return elem_type, shape

def get_valueInfo_dim_type(tensor:Union[onnx.ValueInfoProto, str], model:onnx.ModelProto) -> tuple([int, list]):
    if isinstance(tensor,str): _, tensor = get_value_info(tensor, model)
    elem_type = tensor.type.tensor_type.elem_type
    shape = [dims.dim_value for dims in tensor.type.tensor_type.shape.dim]
    return elem_type, shape

def get_attribute(attribute:onnx.AttributeProto) -> dict:
    attribute_dict = {}
    for attr in attribute:
        temp = 0
        if attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            temp = attr.f
        elif attr.type  == onnx.AttributeProto.AttributeType.INT:
            temp = attr.i
        elif attr.type  == onnx.AttributeProto.AttributeType.STRING:
            temp = attr.s
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            temp = attr.t
        elif attr.type == onnx.AttributeProto.AttributeType.GRAPH:
            temp = attr.g
        elif attr.type == onnx.AttributeProto.AttributeType.SPARSE_TENSOR:
            temp = attr.sparse_tensor
        elif attr.type == onnx.AttributeProto.AttributeType.TYPE_PROTO:
            temp = attr.tp
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            temp = attr.floats
        elif attr.type == onnx.AttributeProto.AttributeType.INTS:
            temp = attr.ints
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            temp = attr.strings
        elif attr.type == onnx.AttributeProto.AttributeType.TENSORS:
            temp = attr.tensors
        elif attr.type == onnx.AttributeProto.AttributeType.GRAPHS:
            temp = attr.graphs
        elif attr.type == onnx.AttributeProto.AttributeType.SPARSE_TENSORS:
            temp = attr.sparse_tensors
        elif attr.type == onnx.AttributeProto.AttributeType.TYPE_PROTOS:
            temp = attr.type_protos
        attribute_dict[attr.name] = temp
    return attribute_dict



def create_tensor_dict(model:Union[onnx.ModelProto,str]) -> tuple([dict, dict]):
    """Create tensor information
        
    Args:
        model : can be onnx.ModelProto or Onnx ModelPath
    Returns:
        dict1  : activative tensor dictionary (operator input / output)
            {"tensor_name": { "dims" : list , "element_type" : onnx.TensorProto }
        dict2  : static tensor dictionary (operator weight, etc.)
            {"tensor_name": { "dims" : list , "element_type" : onnx.TensorProto }
    """    
    if isinstance(model, str): model = onnx.load(model)
    activative_tensor = {}
        
    static_tensor = {}
    for init in model.graph.initializer: # initializer
        static_tensor[init.name] = {
            "dims" : init.dims,
            "element_type" : init.data_type
            }


    for ipt in model.graph.input: # if ir_version < 3 : input + initializer else ir_version > 3 : input
        if ipt.name not in static_tensor.keys():
            activative_tensor[ipt.name] = {
                "dims" : [a.dim_value for a in ipt.type.tensor_type.shape.dim],
                "element_type" : ipt.type.tensor_type.elem_type,
            }
    
    # activative tensor - (input tensor and output tensor)
    for value in model.graph.value_info:
        activative_tensor[value.name] = {
            "dims" : [a.dim_value for a in value.type.tensor_type.shape.dim],
            "element_type" : value.type.tensor_type.elem_type,
        }
    for output in model.graph.output: # model output tensor
        activative_tensor[output.name] = {
            "dims" : [a.dim_value for a in output.type.tensor_type.shape.dim],
            "element_type" : output.type.tensor_type.elem_type,
        }
        
    for i in (model.graph.node):
        for input_name in i.input:
            if input_name in activative_tensor.keys():
                if "consumer" not in activative_tensor[input_name].keys():
                    activative_tensor[input_name]['consumer'] = [i.name]
                else:
                    activative_tensor[input_name]['consumer'].append(i.name)
            elif input_name in static_tensor.keys():
                if "consumer" not in static_tensor[input_name].keys():
                    static_tensor[input_name]['consumer'] = [i.name]
                else:
                    static_tensor[input_name]['consumer'].append(i.name)
        for output_name in i.output:
            if output_name in activative_tensor.keys():
                if "producer" not in activative_tensor[output_name].keys():
                    activative_tensor[output_name]['producer'] = [i.name]
                else:
                    activative_tensor[output_name]['producer'].append(i.name)
        
    return activative_tensor, static_tensor


def create_operator_list_dict(model:Union[onnx.ModelProto,str], static_tensor:dict)-> tuple([list, dict]):
    """Create node list and dictionary
        
    Args:
        model : can be onnx.ModelProto or Onnx ModelPath
        static_tensor :  static tensor dictionary ; created by `src.tool.util.create_tensor_dict`
    Returns:
        list  :  node list : [str, ]
        dict  :  node dict
            { "node_name": {"output" : [str,], "input" : [str,], 'initializer' : [str,]}}
    """    
    if isinstance(model, str): model = onnx.load(model)
    nodeList = [k.name for k in model.graph.node]
    nodeDict = {}
    for node in model.graph.node: 
        inputs = []
        inits = []
        for name in node.input:
            if name not in static_tensor.keys():
                inputs.append(name)
            else:   
                inits.append(name)
        nodeDict[node.name] = {"output" : node.output, "input" : inputs, 'initializer' : inits}
    
    return nodeList, nodeDict
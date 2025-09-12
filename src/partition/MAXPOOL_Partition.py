from . import *
from onnx import helper

def MAXPOOL_Partition(partition: int, model: onnx.ModelProto, maxpool_node_name: str, concat_node_name: str, split_node_name: str) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)

    # 找 MaxPool 節點
    maxpool_node = None
    maxpool_index = 0
    maxPoolNameList = []  
    
    for i, node in enumerate(model.graph.node):
        if node.name == maxpool_node_name:
            maxpool_node = node
            maxpool_index = i
            break

    if maxpool_node is None:
        raise ValueError(f"MaxPool node {maxpool_node_name} not found")

    # 推斷輸入 shape，找 channel 數（第 1 維）
    tensor_idx, input_vi = util.get_value_info(tensor_name=maxpool_node.input[0], model=model)
    elem_type = input_vi.type.tensor_type.elem_type
    shape = [dim.dim_value for dim in input_vi.type.tensor_type.shape.dim]
    C = shape[1]
    # print(f"Input shape: {shape}")

    number = util.output_shape(partition, C)  # e.g. [4,4,4,4] for 16 channel split into 4

    # print(f"Output shape: {number}")

    # Create Split node
    split_output_names = []
    for i in range(partition):
        name = maxpool_node.input[0] + f"/split_{i}"
        split_output_names.append(name)

    split_node = helper.make_node(
        'Split',
        name=split_node_name,
        inputs=[maxpool_node.input[0]],
        outputs=split_output_names,
        axis=1,
        split=number
    )

    model.graph.node.insert(maxpool_index, split_node)

    # 為每個切片創建新的 MaxPool 節點
    maxpool_attrs = {attr.name: helper.get_attribute_value(attr) for attr in maxpool_node.attribute}
    maxpool_output_names = []
    maxpool_names = []
    for i in range(partition):
        out_name = maxpool_node.output[0] + f"/split_{i}"
        maxpool_output_names.append(out_name)

        new_name = maxpool_node.name + f"_{i}"
        maxpool_names.append(new_name)

        new_node = helper.make_node(
            'MaxPool',
            name=new_name,
            inputs=[split_output_names[i]],
            outputs=[out_name],
            **maxpool_attrs
        )
        model.graph.node.insert(maxpool_index + 1 + i, new_node)


    # 插入 Concat node
    concat_node = helper.make_node(
        'Concat',
        name=concat_node_name,
        inputs=maxpool_output_names,
        outputs=maxpool_node.output,
        axis=1
    )

    model.graph.node.insert(maxpool_index + 1 + partition, concat_node)

    # 移除原始 MaxPool
    model.graph.node.remove(maxpool_node)

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    return model, maxpool_names

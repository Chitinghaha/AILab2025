from . import *
from onnx import helper, shape_inference

def LRN_Partition(partition: int, model: onnx.ModelProto, lrn_node_name: str, concat_node_name: str, split_node_name: str) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)

    # 找 lrn 節點
    lrn_node = None
    lrn_index = 0
    lrnNameList = []  
    
    for i, node in enumerate(model.graph.node):
        if node.name == lrn_node_name:
            lrn_node = node
            lrn_index = i
            break

    if lrn_node is None:
        raise ValueError(f"lrn node {lrn_node_name} not found")

    # 推斷輸入 shape，找 W（第 3 維）
    tensor_idx, input_vi = util.get_value_info(tensor_name=lrn_node.input[0], model=model)
    elem_type = input_vi.type.tensor_type.elem_type
    shape = [dim.dim_value for dim in input_vi.type.tensor_type.shape.dim]
    W = shape[3]
    # print(f"Input shape: {shape}")

    number = util.output_shape(partition, W)  # e.g. [14,14,14,14] 分割 W

    # print(f"Output shape: {number}")

    # Create Split node (切 W 維度，所以 axis=3)
    split_output_names = []
    for i in range(partition):
        name = lrn_node.input[0] + f"/split_{i}"
        split_output_names.append(name)

    split_node = helper.make_node(
        'Split',
        name=split_node_name,
        inputs=[lrn_node.input[0]],
        outputs=split_output_names,
        axis=3,
        split=number
    )

    model.graph.node.insert(lrn_index, split_node)

    # 把 split 輸出加入 value_info (形狀需修改 W 維)
    for i, name in enumerate(split_output_names):
        new_shape = shape.copy()
        new_shape[3] = number[i]
        new_vi = helper.make_tensor_value_info(
            name,
            elem_type,
            new_shape
        )
        model.graph.value_info.append(new_vi)

    # 為每個切片創建新的 lrn 節點，並加入輸出 value_info
    lrn_attrs = {attr.name: helper.get_attribute_value(attr) for attr in lrn_node.attribute}
    lrn_output_names = []
    lrn_names = []
    for i in range(partition):
        out_name = lrn_node.output[0] + f"/split_{i}"
        lrn_output_names.append(out_name)

        new_name = lrn_node.name + f"_{i}"
        lrn_names.append(new_name)

        new_node = helper.make_node(
            'LRN',
            name=new_name,
            inputs=[split_output_names[i]],
            outputs=[out_name],
            **lrn_attrs
        )
        model.graph.node.insert(lrn_index + 1 + i, new_node)

        # 新增輸出 value_info，假設形狀與 split 輸出相同
        new_vi = helper.make_tensor_value_info(
            out_name,
            elem_type,
            [shape[0], shape[1], shape[2], number[i]]
        )
        model.graph.value_info.append(new_vi)

    # 插入 Concat node
    concat_node = helper.make_node(
        'Concat',
        name=concat_node_name,
        inputs=lrn_output_names,
        outputs=lrn_node.output,
        axis=3
    )
    model.graph.node.insert(lrn_index + 1 + partition, concat_node)

    # 移除原始 lrn
    model.graph.node.remove(lrn_node)

    # 再推斷形狀、檢查模型
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    return model, lrn_names

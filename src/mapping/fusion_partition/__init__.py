import numpy as np
import onnx
from onnx import numpy_helper

from src.tool.util import output_shape, partition_numpy_kernel, partition_output, get_value_info

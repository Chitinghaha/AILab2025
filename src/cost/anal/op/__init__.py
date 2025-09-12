import onnx
import src.config as config
from src.tool.util import get_value_info, check_tensor_initilizer, get_attribute, get_initilizer
import math
from src.structure import DATA_SIZE_DTYPE
from typing import Optional
from src.management import tool
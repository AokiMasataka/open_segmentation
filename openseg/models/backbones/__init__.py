from openbacks.builder import BACKBONES as OPENBACKS
from ..builder import BACKBONES


for key, value in OPENBACKS._module_dict.items():
    BACKBONES._module_dict[key] = value
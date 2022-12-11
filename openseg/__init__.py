from .core import *
from .dataset import *
from .engin import *
from .models import *
from .utils import *
from .inference import InferenceSegmentor

VERSION = (1, 0, 0)
__version__ = '.'.join(map(str, VERSION))

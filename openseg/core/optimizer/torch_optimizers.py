from inspect import getmembers, isclass
from torch import optim
from ..builder import OPTIMIZERS


optimizer_members = getmembers(object=optim, predicate=isclass)

for optimizer_member in optimizer_members:
    OPTIMIZERS.register_module(module=optimizer_member[1], module_name=None)

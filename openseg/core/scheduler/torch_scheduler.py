from inspect import getmembers, isclass
from torch.optim import lr_scheduler
# from timm import scheduler
from ..builder import SCHEDULERS


@SCHEDULERS.register_module
class DummyScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


members = getmembers(object=lr_scheduler, predicate=isclass)
# members += getmembers(object=scheduler, predicate=isclass)

for member in members:
    SCHEDULERS.register_module(module=member[1], module_name=None)

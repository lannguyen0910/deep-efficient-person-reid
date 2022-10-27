from .cuhk03 import *
from .market import *
from .aihub import *
from .closed_dataset import *
from .image_dataset import *

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'aihub' : AIHUB,
    'closed' : ClosedDataset,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

from lib.optimizer.optimizer_1 import Optimizer1
from torch.optim import SGD, Adam


def get_optimizer(name, params, lr, momentum, weight_decay):
    name = name.lower()
    if name == 'optimizer1':
        return Optimizer1(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print('optimizer only support optimizer1, adam, sgd, got: ', name)
        raise ValueError

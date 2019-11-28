import random
import numpy as np
import torch


def seed_torch(seed=0):
    """
    for deterministic
    """
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_random_state():
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all()

    return {
        'random_state': random_state,
        'numpy_state': numpy_state,
        'torch_state': torch_state,
        'torch_cuda_state': torch_cuda_state
    }


def set_random_state(states):
    random.setstate(states['random_state'])
    np.random.set_state(states['numpy_state'])
    torch.set_rng_state(states['torch_state'])
    torch.cuda.set_rng_state_all(states['torch_cuda_state'])

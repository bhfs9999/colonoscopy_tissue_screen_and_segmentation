import torch
from torch.optim import Optimizer

required = object()


class Optimizer1(Optimizer):
    """Implements Neural Optimizer Search's Optimizer_1 for PyTorch
    """

    def __init__(self, params, lr=required, momentum=0.99, dampening=0,
                 weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super(Optimizer1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Optimizer1, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                # Update rule: g * e(sign(g)*sign(m))
                d_p = d_p.mul(torch.exp(torch.sign(d_p)*torch.sign(buf)))

                p.data.add_(-group['lr'], d_p)

        return loss

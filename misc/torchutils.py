import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.optim import lr_scheduler
from typing import Iterable, Set, Tuple
import logging
import os
logger = logging.getLogger('base')

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    if seg.ndim == 4:
        seg = seg.squeeze(dim=1)
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch']//args['sheduler']['n_steps']
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def save_network(opt, epoch, cd_model, optimizer, F1, is_best_model):
    cd_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_current.pth')
    best_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_best.pth')

    # Save CD model pareamters
    network = cd_model
    if isinstance(cd_model, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()

    opt_state = {'epoch': epoch,
                 'model': state_dict,
                 'F1': F1,
                 'optimizer': optimizer.state_dict()}
    
    if is_best_model:
        torch.save(opt_state, best_path)
    else:
        torch.save(opt_state, cd_path)

def save_network1(opt, epoch, cd_model, optimizer, F1, is_best_model):
    cd_path = os.path.join(
        opt['path_cd']['checkpoint'], f'cd_model_{epoch}-{F1}.pth')
    best_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_best.pth')

    # Save CD model pareamters
    network = cd_model
    if isinstance(cd_model, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()

    opt_state = {'epoch': epoch,
                 'model': state_dict,
                 'F1': F1,
                 'optimizer': optimizer.state_dict()}
    
    torch.save(opt_state, cd_path)

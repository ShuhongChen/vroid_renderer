


try:
    import _util.util_v1 as uutil
except:
    pass


try:
    import torch
    import torch.nn as nn
except:
    pass

try:
    import torchvision as tv
    import torchvision.transforms as TT
    import torchvision.transforms.functional as TF
except:
    pass

try:
    import pytorch_lightning as pl
    import pytorch_lightning.strategies as _
    import torchmetrics
except:
    pass

try:
    import torch_fidelity
except:
    pass

try:
    import wandb
except:
    pass

try:
    import kornia
except:
    pass

try:
    import einops
    from einops.layers import torch as _
except:
    pass

try:
    import cupy
except:
    pass

try:
    from addict import Dict
except:
    Dict = dict


#################### UTILITIES ####################


import contextlib, threading
@contextlib.contextmanager
def torch_seed(seed):
    _torch_seed_lock.acquire()
    state = torch.get_rng_state()
    was_det = torch.backends.cudnn.deterministic
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    elif seed==None:
        seed = torch.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.backends.cudnn.deterministic = was_det
        _torch_seed_lock.release()
_torch_seed_lock = threading.Lock()

def torch_seed_all(seed):
    _torch_seed_lock.acquire()
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    elif seed==None:
        seed = torch.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    _torch_seed_lock.release()
    return


# @cupy.memoize(for_each_device=True)
def cupy_launch(func, kernel):
    return cupy.cuda.compile_with_cache(kernel).get_function(func)

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

def default_collate(items, device=None):
    return to(Dict(torch.utils.data.dataloader.default_collate(items)), device)

def to(x, device):
    if device is None:
        return x
    if issubclass(x.__class__, dict):
        return Dict({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k,v in x.items()
        })
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    assert 0, 'data not understood'


try:
    class IdentityModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            return
        def forward(self, x, *args, **kwargs):
            return x
    class Tanh10Module(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            return
        def forward(self, x, *args, **kwargs):
            return 10*torch.tanh(x/10)
except: pass


def tsuma(t, f=' 2.04f'):
    with torch.no_grad():
        s = f' shape ({",".join([str(i) for i in t.shape])})\n'
        return s+str(uutil.Table([
            # ['shape::l', ('('+','.join([str(i) for i in t.shape])+')', 'r'), ],
            ['min::l', (t.min().item(), f'r:{f}'), ' ', 'std::l', (t.std().item(), f'r:{f}'), ],
            ['mean::l', (t.mean().item(), f'r:{f}'), ' ', 'norm::l', (t.norm().item(), f'r:{f}'), ],
            ['max::l', (t.max().item(), f'r:{f}'), ],
            # ['std::l', (t.std().item(), f'r:{f}'), ],
            # ['norm::l', (t.norm().item(), f'r:{f}'), ],
        ]))+'\n'


#################### METRICS ####################

def binclass_metrics(pred, gt, dims=(1,2,3)):
    assert pred.dtype==torch.bool
    assert gt.dtype==torch.bool
    
    tp = (pred & gt).sum(dims)
    acc = (pred==gt).float().mean(dims)
    pre = tp / pred.sum(dims)
    rec = tp / gt.sum(dims)
    f1 = 2*pre*rec/(pre+rec)

    pzero = pred.sum(dims)==0
    gzero = gt.sum(dims)==0
    pneqg = pzero!=gzero
    pgeqz = pzero & gzero
    f1[pneqg | ((pre==0)&(rec==0))] = 0
    pre[pneqg] = 0
    rec[pneqg] = 0
    pre[pgeqz] = 1
    rec[pgeqz] = 1
    f1[pgeqz] = 1

    return Dict({
        'f1': f1,
        'precision': pre,
        'recall': rec,
        'accuracy': acc,
    })





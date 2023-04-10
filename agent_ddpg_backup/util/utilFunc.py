import numpy as np

def w_decay(nets, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for net in nets:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if "bias" in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

def min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)
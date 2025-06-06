import torch
import torch.nn.functional as F

def round_confidences(conf_tensor, decimal_places):
    if decimal_places ==0:
        return conf_tensor
    factor = 10 ** decimal_places
    return torch.round(conf_tensor * factor) / factor



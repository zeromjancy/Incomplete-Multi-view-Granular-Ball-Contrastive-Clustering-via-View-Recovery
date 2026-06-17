import torch
from torch.nn.functional import one_hot
import numpy as np
from granular.base import GranularBall, GBList, contain_same_sample



def relation_of_views_gblists(view0: GBList, view1: GBList, t=0.1):
    n0, n1 = len(view0), len(view1)
    mask = np.zeros((n0, n1), dtype=np.float32)
    for i in range(n0):
        set0 = set(view0[i].indices)
        for j in range(n1):
            set1 = set(view1[j].indices)
            sub_set = set0 & set1
            if len(sub_set) / len(set0) > t or len(sub_set) / len(set1) > t:
                mask[i, j] = 1
    return torch.from_numpy(mask).to(view0.data.device)


def relation_of_views_gblists_tensor(view0: GBList, view1: GBList, t=0.2):
    y_parts0 = view0.y_parts
    y_parts1 = view1.y_parts
    num_gb = len(view0)

    num_gb1 = len(view1)

    one_hot0 = one_hot(y_parts0, num_classes=num_gb).float()

    one_hot1 = one_hot(y_parts1, num_classes=num_gb1).float()
    mask = one_hot0.T @ one_hot1
    num_gb_set0 = one_hot0.sum(dim=0).view((-1, 1))
    num_gb_set1 = one_hot1.sum(dim=0).view((1, -1))
    num_gb_min = torch.min(num_gb_set0, num_gb_set1) + 1e-8
    mask = (mask / num_gb_min) > t
    return mask.float()



def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4, device):
    merged_tensor = torch.zeros((n + m, n + m), device=device)

    merged_tensor[:n, :n] = tensor1

    merged_tensor[:n, n:n + m] = tensor2

    merged_tensor[n:n + m, :n] = tensor3

    merged_tensor[n:n + m, n:n + m] = tensor4

    return merged_tensor



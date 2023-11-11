import time
import numpy as np
import torch
from torch.nn.functional import *
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import *

import ipdb

class My_Dropout(nn.Module):
    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x:torch.Tensor):
        if self.training:
            self.mask = (torch.rand(x.shape) > self.p).to(dtype=torch.float32, device=x.device)
            return x * self.mask / (1 - self.p)
        else:
            return x


if __name__ == "__main__":
    my_dropout = My_Dropout(p=0.5)
    nn_dropout = nn.Dropout(p=0.5)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0]])
    print(f"输入：\n{x}")
    output_my_dropout = my_dropout(x)
    output_nn_dropout = nn_dropout(x)
    print(f"My_Dropout输出：\n{output_my_dropout}")
    print(f"nn.Dropout输出：\n{output_nn_dropout}")

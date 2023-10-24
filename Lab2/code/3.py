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

class Model_3_1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        self.activate_fn = relu

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate_fn(x)

        x = self.fc2(x)
        x = self.activate_fn(x)

        x = self.fc3(x)
        x = self.activate_fn(x)
        return x
    

class Model_3_2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        self.activate_fn = sigmoid

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate_fn(x)

        x = self.fc2(x)
        x = self.activate_fn(x)

        x = self.fc3(x)
        x = self.activate_fn(x)
        return x
    

class Model_3_3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        self.activate_fn = tanh

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate_fn(x)

        x = self.fc2(x)
        x = self.activate_fn(x)

        x = self.fc3(x)
        x = self.activate_fn(x)
        return x


if __name__ == "__main__":
    train_MNIST_CLS(Model=Model_3_1)
    train_MNIST_CLS(Model=Model_3_2)
    train_MNIST_CLS(Model=Model_3_3)

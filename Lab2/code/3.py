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
    

class Model_3_4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        self.activate_fn = leaky_relu

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
    print("模型1开始训练，激活函数为relu：")
    train_loss_3_1, test_acc_3_1 = train_MNIST_CLS(Model=Model_3_1) # 激活函数为relu
    print("模型2开始训练，激活函数为sigmoid：")
    train_loss_3_2, test_acc_3_2 = train_MNIST_CLS(Model=Model_3_2) # 激活函数为sigmoid
    print("模型3开始训练，激活函数为tanh：")
    train_loss_3_3, test_acc_3_3 = train_MNIST_CLS(Model=Model_3_3) # 激活函数为tanh
    print("模型4开始训练，激活函数为leaky_relu：")
    train_loss_3_4, test_acc_3_4 = train_MNIST_CLS(Model=Model_3_4) # 激活函数为leaky_relu


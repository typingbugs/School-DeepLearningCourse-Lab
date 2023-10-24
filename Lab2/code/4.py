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

class Model_4_1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.activate_fn = leaky_relu

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate_fn(x)

        x = self.fc2(x)
        x = self.activate_fn(x)
        return x
    

class Model_4_2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.activate_fn = leaky_relu

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activate_fn(x)

        x = self.fc2(x)
        x = self.activate_fn(x)
        return x
    

class Model_4_3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
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
    

class Model_4_4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)
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
    print("模型1开始训练，hidden_size=512，hidden_layer=1 ：")
    train_loss_4_1, test_acc_4_1 = train_MNIST_CLS(Model=Model_4_1) # hidden_size=512, hidden_layer=1
    print("模型2开始训练，hidden_size=1024，hidden_layer=1 ：")
    train_loss_4_2, test_acc_4_2 = train_MNIST_CLS(Model=Model_4_2) # hidden_size=1024, hidden_layer=1
    print("模型3开始训练，hidden_size=512，hidden_layer=2 ：")
    train_loss_4_3, test_acc_4_3 = train_MNIST_CLS(Model=Model_4_3) # hidden_size=512, hidden_layer=2
    print("模型4开始训练，hidden_size=1024，hidden_layer=2 ：")
    train_loss_4_4, test_acc_4_4 = train_MNIST_CLS(Model=Model_4_4) # hidden_size=1024, hidden_layer=2


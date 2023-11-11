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
        

class MNIST_CLS_Model(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    learning_rate = 8e-2
    num_epochs = 10
    for i in np.arange(3):
        dropout_rate = 0.1 + 0.4 * i
        model = MNIST_CLS_Model(num_classes=10, dropout_rate=dropout_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f"dropout_rate={dropout_rate}")
        train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)
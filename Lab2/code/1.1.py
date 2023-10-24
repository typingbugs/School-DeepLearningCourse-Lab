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


class Model_1_1:
    def __init__(self):
        self.linear = My_Linear(in_features=500, out_features=1)
        self.sigmoid = My_Sigmoid()
        self.params = self.linear.params

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params
    
    def train(self):
        for param in self.params:
            param.requires_grad = True
    
    def eval(self):
        for param in self.params:
            param.requires_grad = False

class My_Regression_Dataset(Dataset):
    def __init__(self, train=True):
        data_size = 7000 if train else 3000
        np.random.seed(0)
        x = np.random.random((data_size, 500)) * 0.005
        noise = np.random.randn(data_size) * 1e-7
        y = 0.028 - 0.0056 * x.sum(axis=1) + noise
        y = y.reshape(-1, 1)
        self.data = [[x[i], y[i]] for i in range(x.shape[0])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y

if __name__ == "__main__":
    learning_rate = 5
    num_epochs = 10
    batch_size = 512
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_regression_dataset = My_Regression_Dataset(train=True)
    test_regression_dataset = My_Regression_Dataset(train=False)
    train_dataloader = DataLoader(
        dataset=train_regression_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_regression_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )

    model = Model_1_1().to(device)
    criterion = My_BCELoss()
    optimizer = My_optimizer(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        start_time = time.time()
        for index, (x, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            
            x = x.to(device)
            targets = targets.to(device)
            
            y_pred = model(x)
            
            loss = criterion(y_pred, targets)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        end_time = time.time()
        train_time = end_time - start_time

        model.eval()
        with torch.no_grad():
            total_epoch_acc = 0
            start_time = time.time()
            for index, (x, targets) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                x = x.to(device)
                targets = targets.to(device)
                
                y_pred = model(x)
                total_epoch_acc += (1 - torch.abs(y_pred - targets) / torch.abs(targets)).sum().item()
                
            end_time = time.time()
            test_time = end_time - start_time
            
        avg_epoch_acc = total_epoch_acc / len(test_regression_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss:.10f},",
            f"Used Time: {train_time * 1000:.3f}ms,",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
            f"Used Time: {test_time * 1000:.3f}ms",
        )
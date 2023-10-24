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


class Model_1_2:
    def __init__(self):
        self.fc = My_Linear(in_features=200, out_features=1)
        self.sigmoid = My_Sigmoid()
        self.params = self.fc.parameters()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.fc(x)
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

class My_BinaryCLS_Dataset(Dataset):
    def __init__(self, train=True, num_features=200):
        num_samples = 7000 if train else 3000
        
        x_1 = np.random.normal(loc=-0.5, scale=0.2, size=(num_samples, num_features))
        x_2 = np.random.normal(loc=0.5, scale=0.2, size=(num_samples, num_features))
        
        labels_1 = np.zeros((num_samples, 1))
        labels_2 = np.ones((num_samples, 1))
        
        x = np.concatenate((x_1, x_2), axis=0)
        labels = np.concatenate((labels_1, labels_2), axis=0)
        self.data = [[x[i], labels[i]] for i in range(2 * num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        return x, y
    

if __name__ == "__main__":
    learning_rate = 5e-3
    num_epochs = 10
    batch_size = 512
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_binarycls_dataset = My_BinaryCLS_Dataset(train=True)
    test_binarycls_dataset = My_BinaryCLS_Dataset(train=False)
    train_dataloader = DataLoader(
        dataset=train_binarycls_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_binarycls_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )

    model = Model_1_2().to(device)
    criterion = My_BCELoss()
    optimizer = My_optimizer(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        start_time = time.time()
        for index, (x, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            x = x.to(device)
            targets = targets.to(device).to(dtype=torch.float)
            
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
                
                output = model(x)
                pred = (output > 0.5).to(dtype=torch.long)
                total_epoch_acc += (pred == targets).sum().item()
                
            end_time = time.time()
            test_time = end_time - start_time

        avg_epoch_acc = total_epoch_acc / len(test_binarycls_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss},",
            f"Used Time: {train_time * 1000:.3f}ms,",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
            f"Used Time: {test_time * 1000:.3f}ms",
        )

import time
import numpy as np
import torch
from torch.nn.functional import one_hot, softmax
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import *

import ipdb
        

class Model_1_3:
    def __init__(self, num_classes):
        self.flatten = My_Flatten()
        self.linear = My_Linear(in_features=28 * 28, out_features=num_classes)
        self.params = self.linear.params

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear(x)
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

if __name__ == "__main__":
    learning_rate = 1e-1
    num_epochs = 10
    batch_size = 512
    num_classes = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_mnist_dataset = datasets.MNIST(root="../dataset", train=True, transform=transform, download=True)
    test_mnist_dataset = datasets.MNIST(root="../dataset", train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=train_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )

    model = Model_1_3(num_classes).to(device)
    criterion = My_CrossEntropyLoss()
    optimizer = My_optimizer(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        start_time = time.time()
        for index, (images, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            one_hot_targets = my_one_hot(targets, num_classes=num_classes).to(dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, one_hot_targets)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        end_time = time.time()
        train_time = end_time - start_time

        model.eval()
        with torch.no_grad():
            total_epoch_acc = 0
            start_time = time.time()
            for index, (image, targets) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                image = image.to(device)
                targets = targets.to(device)
                
                outputs = model(image)
                pred = my_softmax(outputs, dim=1)
                total_epoch_acc += (pred.argmax(1) == targets).sum().item()
            
            end_time = time.time()
            test_time = end_time - start_time
        
        avg_epoch_acc = total_epoch_acc / len(test_mnist_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss},",
            f"Used Time: {train_time * 1000:.3f}ms,",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
            f"Used Time: {test_time * 1000:.3f}ms",
        )

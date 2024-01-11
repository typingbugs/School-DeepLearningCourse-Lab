import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import os
import time
from PIL import Image
import pandas as pd

class Vehicle(Dataset):
    def __init__(self, root:str="../dataset", train:bool=True, transform=None):
        root = os.path.join(root, "Vehicles")
        csv_file = os.path.join(root, "train.csv" if train else "test.csv")
        self.data = pd.read_csv(csv_file).to_numpy().tolist()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path)
        label = int(label)
        if self.transform:
            image = self.transform(image)
        return image, label
    

class Haze(Dataset):
    def __init__(self, root:str="../dataset", train:bool=True, transform=None):
        root = os.path.join(root, "Haze")
        split_file = pd.read_csv(os.path.join(root, "split.csv")).to_numpy().tolist()
        self.data = list()
        for img, is_train in split_file:
            if train and int(is_train) == 1:
                self.data.append(img)
            elif not train and int(is_train) == 0:
                self.data.append(img)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name = self.data[index]
        img_path = os.path.join(self.root, "raw/haze", img_name)
        ground_truth_path = os.path.join(self.root, "raw/no_haze", img_name)
        image = Image.open(img_path)
        ground_truth = Image.open(ground_truth_path)
        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)
        return image, ground_truth
    

def train_Vehicle_CLS(model:nn.Module, learning_rate=1e-3, batch_size=64, num_epochs=51):
    num_classes = 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = Vehicle(root="../dataset", train=True, transform=transform)
    test_dataset = Vehicle(root="../dataset", train=False, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=14, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = list()
    test_acc = list()
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        train_tik = time.time()
        for index, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).to(dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, one_hot_targets)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_tok = time.time()

        model.eval()
        with torch.no_grad():
            total_epoch_acc = 0
            test_tik = time.time()
            for index, (image, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
                image = image.to(device)
                targets = targets.to(device)

                outputs = model(image)
                pred = F.softmax(outputs, dim=1)
                total_epoch_acc += (pred.argmax(1) == targets).sum().item()
            test_tok = time.time()

        avg_epoch_acc = total_epoch_acc / len(test_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss:.10f},",
            f"Train Time: {1000 * (train_tok - train_tik):.2f}ms,",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
            f"Test Time: {1000 * (test_tok - test_tik):.2f}ms"
        )
        train_loss.append(total_epoch_loss)
        test_acc.append(avg_epoch_acc)
    print(f"最大显存使用量: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}MiB")
    torch.cuda.reset_peak_memory_stats()
    return train_loss, test_acc


def train_Haze_Removal(model:nn.Module, learning_rate=1e-3, batch_size=64, num_epochs=51):
    num_epochs = 50
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
        ]
    )
    train_dataset = Haze(root="../dataset", train=True, transform=transform)
    test_dataset = Haze(root="../dataset", train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=14, pin_memory=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = list()
    test_loss = list()
    for epoch in range(num_epochs):
        model.train()
        total_epoch_train_loss = 0
        train_tik = time.time()
        for index, (images, ground_truth) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            images = images.to(device)
            ground_truth = ground_truth.to(device)

            outputs = model(images)
            loss = criterion(outputs, ground_truth)
            total_epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_tok = time.time()

        model.eval()
        with torch.no_grad():
            total_epoch_test_loss = 0
            test_tik = time.time()
            for index, (image, ground_truth) in tqdm(enumerate(test_loader), total=len(test_loader)):
                image = image.to(device)
                ground_truth = ground_truth.to(device)

                outputs = model(image)
                loss = criterion(outputs, ground_truth)
                total_epoch_test_loss += loss.item()
            test_tok = time.time()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_train_loss:.10f},",
            f"Train Time: {1000 * (train_tok - train_tik):.2f}ms,",
            f"Test Loss: {total_epoch_test_loss:.10f},",
            f"Test Time: {1000 * (test_tok - test_tik):.2f}ms"
        )
        train_loss.append(total_epoch_train_loss)
        test_loss.append(total_epoch_test_loss)
    print(f"最大显存使用量: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}MiB")
    torch.cuda.reset_peak_memory_stats()
    return train_loss, test_loss
    
    
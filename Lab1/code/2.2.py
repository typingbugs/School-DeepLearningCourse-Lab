import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import ipdb


class Model_2_2(nn.Module):
    def __init__(self):
        super(Model_2_2, self).__init__()
        self.linear = nn.Linear(1, 1, dtype=torch.float64)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


class My_Dataset(Dataset):
    def __init__(self, data_size=1000000):
        np.random.seed(0)
        x = 2 * np.random.rand(data_size, 1)
        noise = 0.2 * np.random.randn(data_size, 1)
        y = 4 - 3 * x + noise
        self.min_x, self.max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        x = (x - self.min_x) / (self.max_x - self.min_x)
        y = (y - min_y) / (max_y - min_y)
        self.data = [[x[i][0], y[i][0]] for i in range(x.shape[0])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y


learning_rate = 5e-2
num_epochs = 10
batch_size = 1024
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = My_Dataset()
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=14,
    pin_memory=True,
)

model = Model_2_2().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_epoch_loss = 0
    total_epoch_pred = 0
    total_epoch_target = 0
    for index, (x, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()

        x = x.to(device)
        targets = targets.to(device)

        x = x.unsqueeze(1)
        targets = targets.unsqueeze(1)
        y_pred = model(x)
        loss = criterion(y_pred, targets)
        total_epoch_loss += loss.item()
        total_epoch_target += targets.sum().item()
        total_epoch_pred += y_pred.sum().item()

        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_epoch_loss}, Acc: {1 - abs(total_epoch_pred - total_epoch_target) / total_epoch_target}"
    )

with torch.no_grad():
    test_data = (np.array([[2]]) - dataset.min_x) / (dataset.max_x - dataset.min_x)
    test_data = Variable(
        torch.tensor(test_data, dtype=torch.float64), requires_grad=False
    ).to(device)
    predicted = model(test_data).to("cpu")
    print(
        f"Model weights: {model.linear.weight.item()}, bias: {model.linear.bias.item()}"
    )
    print(f"Prediction for test data: {predicted.item()}")

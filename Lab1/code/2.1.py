import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import ipdb


class My_BCELoss:
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = -torch.mean(
            target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)
        )
        return loss


class My_optimizer:
    def __init__(self, params: list[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()


class My_Linear:
    def __init__(self, input_feature: int, output_feature: int):
        self.weight = torch.randn(
            (output_feature, input_feature), requires_grad=True, dtype=torch.float32
        )
        self.bias = torch.randn(1, requires_grad=True, dtype=torch.float32)
        self.params = [self.weight, self.bias]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = torch.matmul(x, self.weight.T) + self.bias
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params


class Model:
    def __init__(self):
        self.linear = My_Linear(1, 1)
        self.params = self.linear.params

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params


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
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True
)

model = Model().to(device)
criterion = My_BCELoss()
optimizer = My_optimizer(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_epoch_loss = 0
    total_epoch_pred = 0
    total_epoch_target = 0
    for index, (x, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        x = x.to(device).to(dtype=torch.float32)
        targets = targets.to(device).to(dtype=torch.float32)
        x = x.unsqueeze(1)
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

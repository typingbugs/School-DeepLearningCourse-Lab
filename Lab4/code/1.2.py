from utils import *
import ipdb

class Model_Vehicle_CLS_1_2(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class Model_Haze_Removal_1_2(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    model = Model_Vehicle_CLS_1_2()
    train_Vehicle_CLS(model=model, learning_rate=4e-4, batch_size=64)

    model = Model_Haze_Removal_1_2()
    train_Haze_Removal(model=model, learning_rate=5e-3, batch_size=16)

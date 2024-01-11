from utils import *

class Model_Vehicle_CLS_1_3_1(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_3_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model_Vehicle_CLS_1_3_2(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_3_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model_Vehicle_CLS_1_3_3(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_3_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model_Vehicle_CLS_1_3_4(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_3_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class Model_Haze_Removal_1_3_1(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_3_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Model_Haze_Removal_1_3_2(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_3_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
        )
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Model_Haze_Removal_1_3_3(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_3_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=7, padding=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Model_Haze_Removal_1_3_4(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_3_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=9, padding=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

if __name__ == "__main__":
    num_epochs = 61
    learning_rate = 2e-4
    batch_size = 256
    models = [
        Model_Vehicle_CLS_1_3_1, 
        Model_Vehicle_CLS_1_3_2,
        Model_Vehicle_CLS_1_3_3,
        Model_Vehicle_CLS_1_3_4,
    ]
    for i in range(4):
        model = models[i]()
        print(f"卷积层层数={i + 1}")
        train_loss, test_acc = train_Vehicle_CLS(model=model, learning_rate=learning_rate,
                                                 batch_size=batch_size, num_epochs=num_epochs)
        print()

    num_epochs = 61
    learning_rate = 8e-3
    batch_size = 64
    models = [
        Model_Haze_Removal_1_3_1, 
        Model_Haze_Removal_1_3_2, 
        Model_Haze_Removal_1_3_3, 
        Model_Haze_Removal_1_3_4, 
    ]
    for i in range(4):
        model = models[i]()
        print(f"卷积核大小={3 + 2 * i}")
        train_loss, test_loss = train_Haze_Removal(model=model, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
        print()
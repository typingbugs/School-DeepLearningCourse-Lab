from utils import *

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model_Vehicle_CLS_3(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.conv2 = BasicResidualBlock(in_channels=64, out_channels=64)
        self.conv3 = BasicResidualBlock(in_channels=64, out_channels=64)
        self.conv4 = BasicResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.conv5 = BasicResidualBlock(in_channels=128, out_channels=128)
        self.conv6 = BasicResidualBlock(in_channels=128, out_channels=256, stride=2)
        self.conv7 = BasicResidualBlock(in_channels=256, out_channels=256)
        self.conv8 = BasicResidualBlock(in_channels=256, out_channels=512, stride=2)
        self.conv9 = BasicResidualBlock(in_channels=512, out_channels=512)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    num_epochs = 61
    learning_rate = 15e-5
    batch_size = 512
    model = Model_Vehicle_CLS_3(num_classes=3)
    train_loss, test_acc = train_Vehicle_CLS(model=model, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
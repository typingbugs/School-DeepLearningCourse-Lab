from utils import *
import ipdb


class My_Conv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int=0, bias=True):
        super(My_Conv2d, self).__init__()
        self.has_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, requires_grad=True, dtype=torch.float32))

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = F.unfold(x, kernel_size=self.kernel_size)
        x = x.permute(0, 2, 1).contiguous()
        weight_unfold = self.weight.view(self.out_channels, -1).t()
        x = torch.matmul(x, weight_unfold)
        if self.has_bias:
            x += self.bias
        output_height = input_height + 2 * self.padding - self.kernel_size + 1
        output_width = input_width + 2 * self.padding - self.kernel_size + 1
        x = x.view(batch_size, output_height, output_width, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return x
    

class Model_Vehicle_CLS_1_1(nn.Module):
    def __init__(self, num_classes=3):
        super(Model_Vehicle_CLS_1_1, self).__init__()
        self.conv1 = My_Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = My_Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class Model_Haze_Removal_1_1(nn.Module):
    def __init__(self):
        super(Model_Haze_Removal_1_1, self).__init__()
        self.conv1 = My_Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = My_Conv2d(in_channels=16, out_channels=48, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = My_Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    model = Model_Vehicle_CLS_1_1()
    train_Vehicle_CLS(model=model, learning_rate=4e-4, batch_size=256)
    
    model = Model_Haze_Removal_1_1()
    train_Haze_Removal(model=model, learning_rate=5e-3, batch_size=16)

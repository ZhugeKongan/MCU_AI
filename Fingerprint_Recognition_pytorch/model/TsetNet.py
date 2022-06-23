import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv3x3BNReLU(in_channels,out_channels,stride):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class TsetNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TsetNet, self).__init__()
        self.conv1 = Conv3x3BNReLU(3, 8, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = Conv3x3BNReLU(8, 16, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = Conv3x3BNReLU(16, 16, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = Conv3x3BNReLU(16, 32, 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = Conv3x3BNReLU(32, 48, 1)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc=nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.maxpool4(self.conv4(x))
        x = self.conv5(x)
        # print(x.size())
        x = x.reshape(int(x.size(0)), -1)
        x=self.fc(x)
        return x

if __name__=='__main__':
    model = TsetNet()
    # model = torchvision.models.MobileNetV2()
    print(model)
    input = torch.randn(1, 3, 64, 64)
    save_path = 'test__params.onnx'
    torch.onnx.export(model, input, save_path, export_params=True, verbose=False, opset_version=10)

    out = model(input)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, in_planes, out_planes, expansion, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#
#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
#                                stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, groups=planes,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
#                                stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)
#
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1,
#                           stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layers = self._make_layers(in_planes=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        # print(x.size())#([128, 1, 128, 128])
        out = F.relu(self.bn1(self.conv1(x)))
        out =self.maxpool(out)
        #out = self.layers(out)
        # print(out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # print(out.size())
        out = F.max_pool2d(out,(16, 16))
        # print(out.size())
        out = out.reshape(int(out.size(0)), -1)
        out = self.linear(out)
        return out

if __name__ == '__main__':
    net = MobileNetV2()
    x = torch.randn(1,1,64,64)
    y = net(x)
    print(y.size())
    from thop import profile

    input = torch.randn(1,1,128,128)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('  + Number of params: %.3fG' % (flops / 1e9))
    print('flops: ', flops, 'params: ', params)


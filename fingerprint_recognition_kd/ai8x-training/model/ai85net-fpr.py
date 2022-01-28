###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
FaceID network for AI85/AI86

Optionally quantize/clamp activations
"""
import torch.nn as nn

import ai8x


class AI85FPRNet(nn.Module):
    """
    Simple FaceNet Model
    """
    def __init__(
            self,
            num_classes=None,  # pylint: disable=unused-argument
            num_channels=3,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=False, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        #self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 # padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=False, **kwargs)
        self.avgpool = ai8x.AvgPool2d((4, 4))
        #self.fc = ai8x.Linear(512, num_classes, bias=True, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        #print(x.size())
        x = self.avgpool(x)
        #x=x.view(x.size(0),512)
        #x=self.fc(x)
        return x


def ai85fprnet(pretrained=False, **kwargs):
    """
    Constructs a FaceIDNet model.
    """
    assert not pretrained
    return AI85FPRNet(**kwargs)


models = [
    {
        'name': 'ai85fprnet',
        'min_input': 1,
        'dim': 3,
    },
]

import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class res(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(res, self).__init__()
        self.out_channels = out_planes
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(out_planes, out_planes, kernel_size=3, stride=1, padding=2, dilation=2)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(out_planes, out_planes, kernel_size=3, stride=1, padding=2, dilation=2)
                )

        self.ConvLinear = BasicConv(2*out_planes, 1, kernel_size=1, stride=1,relu=False,bn=False)

    def forward(self,x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1, x2), 1)
        out = self.ConvLinear(out)

        return out

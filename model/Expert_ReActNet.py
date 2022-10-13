from contextlib import suppress
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

if __name__ == '__main__':
    from binaryfunction import *
else:
    from .binaryfunction import *

input_list = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
output_list = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
stride_list = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)

    def forward(self, input):
        if self.training:
            input = SignSTE.apply(input)
            self.weight_bin_tensor = SignWeight.apply(self.weight)
        else:
            # We clone the input here because it causes unexpected behaviors
            # to edit the data of `input` tensor.
            input = input.clone()
            input.data = input.sign()
            # Even though there is a UserWarning here, we have to use `new_tensor`
            # rather than the "recommended" way
            self.weight_bin_tensor = self.weight.new_tensor(self.weight.sign())
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weight), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        self.weight_bin_tensor = self.weight_bin_tensor * scaling_factor
        # 1. The input of binary convolution shoule be only +1 or -1,
        #    so instead of padding 0 automatically, we need pad -1 by ourselves
        # 2. `padding` of nn.Conv2d is always a tuple of (padH, padW),
        #    while the parameter of F.pad should be (padLeft, padRight, padTop, padBottom)

        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = binaryconv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2


class reactnet(nn.Module):
    def __init__(self, num_classes=1000, num_experts=3, returns_feat=True):
        super(reactnet, self).__init__()
        self.conv1 = firstconv3x3(3, 32, 2)
        self.feature = nn.ModuleList()
        self.num_experts = num_experts
        self.returns_feat = returns_feat
        self.requires_target = False
        self.shared_backbone = self._make_layer(BasicBlock,
                                                in_planes=input_list[:7],
                                                out_planes=output_list[:7],
                                                strides=stride_list[:7])
        self.experts = nn.ModuleList([
            self._make_layer(BasicBlock,
                             in_planes=input_list[7:],
                             out_planes=output_list[7:],
                             strides=stride_list[7:])
            for _ in range(num_experts)
        ])
        self.linears = nn.ModuleList([nn.Linear(1024, num_classes) for _ in range(num_experts)])

    def _make_layer(self, block, in_planes, out_planes, strides):
        assert len(in_planes) - len(out_planes) == 0
        assert len(out_planes) - len(strides) == 0
        layers = []
        for i in range(len(in_planes)):
            layers.append(block(in_planes[i], out_planes[i], strides[i]))
        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        x = (self.experts[ind])(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        self.feat.append(x)
        x = (self.linears[ind])(x)
        return x

    def forward(self, x):
        outs = []
        self.feat = []
        out = self.conv1(x)
        out = self.shared_backbone(out)

        for ind in range(self.num_experts):
            outs.append(self._separate_part(out, ind))
        self.logits = outs
        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out,
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(self.logits, dim=1)

            }
        else:
            return final_out


class Classifier(nn.Module):
    def __init__(self, feat_in=1024, num_classes=1000):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)
        init.kaiming_normal_(self.fc.weight)
        self.pool1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = reactnet()
    print(model)
    inp = torch.randn(32, 3, 224, 224)
    out = model(inp)
    print(out)

"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, use_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=use_bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dropout=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)   # nn.ModuleList(layers)
    
    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m
    
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)
        f0 = out

        if preact:
            out, f1_pre = self.layer1(out)
        else:
            out = self.layer1(out)
        f1 = out

        if preact:
            out, f2_pre = self.layer2(out)
        else:
            out = self.layer2(out)
        f2 = out

        if preact:
            out, f3_pre = self.layer3(out)
        else:
            out = self.layer3(out)
        f3 = out

        if preact:
            out, f4_pre = self.layer4(out)
        else:
            out = self.layer4(out)
        f4 = out

        out = self.avgpool(out)
        feats = out.view(out.size(0), -1)
        f5 = feats
        # x = F.dropout(x, 0.5, self.training, True)
        out = self.fc(feats)
        
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out, feats]
            else:
                return [f0, f1, f2, f3, f4, f5], out, feats
        else:
            return out, feats

    def load_pretrained(self, fn):
        checkpoint = torch.load(fn)
        state_dict = self.state_dict()
        for k in state_dict.keys():
            state_dict[k] = checkpoint['state_dict'][k]
        self.load_state_dict(state_dict)
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet18'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet34'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet50'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet101'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet152'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model


def create_model(arch, pretrained=True, **kwargs):
    return eval(arch)(pretrained=pretrained, **kwargs)



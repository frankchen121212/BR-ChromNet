#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-12-25 15:22
# @Author  : Siyuan Chen
# @Site    :
# @File    : ensembled_discriminator.py
# @Software: PyCharm
from collections import namedtuple
import torch
import torch.nn as nn

import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


def get_resnet50(OUTPUT_DIM):
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])
    pretrained_model = models.resnet50(pretrained=True)

    IN_FEATURES = pretrained_model.fc.in_features
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_model.fc = fc
    model = ResNet(resnet50_config, OUTPUT_DIM)
    model.load_state_dict(pretrained_model.state_dict())

    return model

def get_efficientnetV2(OUTPUT_DIM):
    model = models.efficientnet_v2_s(pretrained=True)
    IN_FEATURES = model.classifier[1].in_features
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    model.classifier[1] = fc
    return model


def get_convnext(OUTPUT_DIM):
    model = models.convnext_small(pretrained=True)
    IN_FEATURES = model.classifier[2].in_features
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    model.classifier[2] = fc
    return model

def get_vit(OUTPUT_DIM):
    model = models.vit_b_16(pretrained=True)
    IN_FEATURES = model.heads[0].in_features
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    model.heads[0] = fc
    return model

def get_vgg16(OUTPUT_DIM):
    model = models.vgg16(pretrained=True)
    IN_FEATURES = model.classifier[-1].in_features
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    model.classifier[-1] = fc
    return model

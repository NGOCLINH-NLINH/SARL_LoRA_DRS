# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List, Tuple
from backbone.utils.k_winners import KWinners2d


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SparseBasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, kw_percent_on=0.1, local=False, relu=False, apply_kw=True) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.apply_kw = apply_kw

        self.kw1 = KWinners2d(
            channels=planes,
            percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0,
            local=local,
            relu=relu,
        )

        self.kw2 = KWinners2d(
            channels=planes,
            percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0,
            local=local,
            relu=relu,
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """

        out = self.bn1(self.conv1(x))
        if self.apply_kw:
            out = self.kw1(out)
        else:
            out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.apply_kw:
            out = self.kw2(out)
        else:
            out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    """
    Sparse ResNet network architecture with k-WTA activations. Designed for complex datasets.
    """

    def __init__(self, block: SparseBasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, kw_percent_on=(0.1, 0.1, 0.1, 0.1), local=False, relu=False, apply_kw=(1, 1, 1, 1)) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(SparseResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.num_blocks = num_blocks

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, kw_percent_on=kw_percent_on[0], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, kw_percent_on=kw_percent_on[1], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, kw_percent_on=kw_percent_on[2], local=local, relu=relu, apply_kw=apply_kw[0])
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, kw_percent_on=kw_percent_on[3], local=local, relu=relu, apply_kw=apply_kw[0])
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

        self.classifier = self.linear

    def _make_layer(self, block: SparseBasicBlock, planes: int,
                    num_blocks: int, stride: int, kw_percent_on: float, local: bool, relu:bool, apply_kw:bool) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kw_percent_on, local, relu, apply_kw))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_activations=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        activations = {}
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        activations['feat'] = out
        out = self.linear(out)
        if return_activations:
            return out, activations
        else:
            return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        feat = self._features(x)
        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feat, out

    def extract_features(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(out)  # 64, 32, 32
        feat2 = self.layer2(feat1)  # 128, 16, 16
        feat3 = self.layer3(feat2)  # 256, 8, 8
        feat4 = self.layer4(feat3)  # 512, 4, 4
        out = avg_pool2d(feat4, feat4.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return (feat1, feat2, feat3, feat4), out

    def get_features_only(self, x: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """

        feat = relu(self.bn1(self.conv1(x)))

        if feat_level > 0:
            feat = self.layer1(feat)  # 64, 32, 32
        if feat_level > 1:
            feat = self.layer2(feat)  # 128, 16, 16
        if feat_level > 2:
            feat = self.layer3(feat)  # 256, 8, 8
        if feat_level > 3:
            feat = self.layer4(feat)  # 512, 4, 4
        return feat

    def predict_from_features(self, feats: torch.Tensor, feat_level: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param feats: input tensor (batch_size, *input_shape)
        :param feat_level: resnet block
        :return: output tensor (??)
        """

        out = feats

        if feat_level < 1:
            out = self.layer1(out)  # 64, 32, 32
        if feat_level < 2:
            out = self.layer2(out)  # 128, 16, 16
        if feat_level < 3:
            out = self.layer3(out)  # 256, 8, 8
        if feat_level < 4:
            out = self.layer4(out)  # 512, 4, 4

        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def sparse_resnet18(nclasses: int, nf: int=64, kw_percent_on=(0.9, 0.9, 0.9, 0.9), local=False, relu=False, apply_kw=(1, 1, 1, 1)) -> SparseResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], nclasses, nf, kw_percent_on, local, relu, apply_kw)



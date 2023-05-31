import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from utils.geometry import rot6d_to_rotmat
import os
import os.path as osp
import torch.nn.functional as F
from model.spin import Regressor, hmr, HMR
"""
To use adaptator, we will try two kinds of schemes.
"""


def gn_helper(planes):
    if 0:
        return nn.BatchNorm2d(planes)
    else:
        return nn.GroupNorm(32 // 8, planes)


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            # input_size=128,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer=gn_helper, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class VIBE(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        # Resnet
        norm_layer = gn_helper
        self.inplanes = 64
        super(VIBE, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(norm_layer, block, 64, layers[0])
        self.layer2 = self._make_layer(norm_layer, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(norm_layer, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(norm_layer, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # GRU
        #####
        # 想法是在底层训练时没有GRU，在高层训练时加上GRU，但是不统一
        # 全都改成视频帧，多帧输入，统一都有GRU
        #########
        self.encoder = TemporalEncoder()

        # Regressor
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)  # 初始化使通过网络层时，输入和输出的方差相同，包括前向传播和后向传播
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)  # xavier_uniform_ 均匀分布方式给激活函数初始化
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # for m in self.modules():来自源程序HMR
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # 读smpl初始平均参数
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, norm_layer, block, planes, blocks, stride=1):  # resnet
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_layer, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, need_feature=False, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        features = []

        batch_size, seqlen, nc, h, w = x.shape  # 1

        x1 = x.reshape(-1, nc, h, w) # [16,3,224,224]
        x = x1.to(torch.float32)
        x = self.conv1(x)  # 16,64,112,112
        features.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 16,64,56,56

        x1 = self.layer1(x)  # 16，256，56，56
        features.append(x1)
        x2 = self.layer2(x1)  # 16，512，28，28
        features.append(x2)
        x3 = self.layer3(x2)  # 16，1024，14，14
        features.append(x3)
        x4 = self.layer4(x3)  # 16，2048，7，7
        features.append(x4)

        xf = self.avgpool(x4)  # 16，2048，1，1
        xf = xf.view(xf.size(0), -1)  # 16，2048

        features.append(xf)

        # GRU
        xf = xf.reshape(batch_size, seqlen, -1)
        xg = self.encoder(xf)  # [8,16,2048]
        xr = xg.reshape(-1, xg.size(-1))  # [128,2048]

        features.append(xr)

        # Regressor
        # 注意此batch_new非上面的batch_size，他是通过GRU后resize的第一维度的大小,在这里是128
        batch_new = xr.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_new, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_new, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_new, -1)

        pred_pose = init_pose  # [128,144]
        pred_shape = init_shape  # [128,10]
        pred_cam = init_cam  # [128,3]
        for i in range(n_iter):
            xc = torch.cat([xr, pred_pose, pred_shape, pred_cam], 1)  # 1.[128,2205] 2/3.[128,1024]
            xc = self.fc1(xc)  # [128,1024]
            features.append(xc.clone())
            xc = self.drop1(xc)  # [128,1024]
            features.append(xc.clone())
            xc = self.fc2(xc)  # [128,1024]
            features.append(xc.clone())
            xc = self.drop2(xc)  # [128,1024]
            pred_pose = self.decpose(xc) + pred_pose  # [128,144]
            pred_shape = self.decshape(xc) + pred_shape  # [128,10]
            pred_cam = self.deccam(xc) + pred_cam  # [128,3]

        # features.append(xc)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_new, 24, 3, 3)

        if not need_feature:
            pred_rotmat=pred_rotmat.reshape(batch_size, seqlen, pred_rotmat.size(-3),pred_rotmat.size(-2), pred_rotmat.size(-1))
            pred_shape=pred_shape.reshape(batch_size, seqlen, pred_shape.size(-1))
            pred_cam=pred_cam.reshape(batch_size, seqlen, pred_cam.size(-1))
            return pred_rotmat, pred_shape, pred_cam
        else:
            pred_rotmat = pred_rotmat.reshape(batch_size, seqlen, pred_rotmat.size(-3), pred_rotmat.size(-2),
                                              pred_rotmat.size(-1))
            pred_shape = pred_shape.reshape(batch_size, seqlen, pred_shape.size(-1))
            pred_cam = pred_cam.reshape(batch_size, seqlen, pred_cam.size(-1))
            return pred_rotmat, pred_shape, pred_cam, features


def vibe(smpl_mean_params, pretrained=False, **kwargs):
    """ Constructs the VIBE model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VIBE(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    if pretrained:  # resnet是否预训练
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(), strict=False)
    return model

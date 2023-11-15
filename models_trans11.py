import math
import torch
import torch.nn as nn
from torch.nn import init

# 含6个类，SegDecNet为模型，其他为工具类

BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        init.uniform_(self.weight, 0, 1)
        init.zeros_(self.bias)


class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


def _conv_block(in_chanels, out_chanels, kernel_size, padding, stride=1):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())


class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(
            torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


# 继承 nn.Module（它本身是一个类并且能够跟踪状态）
class SegDecNet(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels, drop_p):
        super(SegDecNet, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        # 分割层1：卷积（特征图提取）
        self.volume = nn.Sequential(_conv_block(self.input_channels, 32, 3, 1),
                                    _conv_block(32, 64, 3, 1),
                                    nn.MaxPool2d(2),
                                    _conv_block(64, 64, 3, 1),
                                    nn.MaxPool2d(2),
                                    _conv_block(64, 128, 3, 1),
                                    nn.MaxPool2d(2),
                                    _conv_block(128, 256, 3, 1),
                                    )

        # 分割层2：正式分割
        self.seg_mask = nn.Sequential(
            Conv2d_init(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False),
            # nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.Sigmoid()
            # nn.Tanh()
            )  # ？？？

        # 决策层1：卷积（特征图提取）
        self.extractor = nn.Sequential(
                                       # nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1, out_chanels=36, kernel_size=5, padding=2),
                                       # nn.MaxPool2d(kernel_size=2),
                                       # _conv_block(in_chanels=4, out_chanels=8, kernel_size=5, padding=2),
                                       # nn.MaxPool2d(kernel_size=2),
                                       # _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       # nn.MaxPool2d(kernel_size=2),
                                       # _conv_block(in_chanels=32, out_chanels=64, kernel_size=3, padding=1),
                                       )

        # 决策层2：池化
        # self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        # self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        # self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        # self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        # 决策层3：最后的1分类器
        self.fc = nn.Sequential(nn.Dropout(p=drop_p),
                                nn.Linear(in_features=36, out_features=1))

        # ？？？（大概是自定义拼接的反向求导函数，具体使用请ctrl+f搜索本页）
        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        # GPU
        self.device = device

    # torch.ones((1,) = tensor([1.])
    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        # 分割层1：卷积（特征图提取）
        volume = self.volume(input)
        # 分割层2：正式分割
        seg_mask = self.seg_mask(volume)

        # 梯度流控制组件（大概是自定义拼接的反向求导函数）
        seg_mask = self.volume_lr_multiplier_layer(seg_mask, self.volume_lr_multiplier_mask)

        # 决策层1：卷积（特征图提取）
        features = self.extractor(seg_mask)
        # 决策层2：池化
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        # global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        # global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        # global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)

        # reshape，并自定义分割池化的反向求导函数
        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        # global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        # global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        # print(global_max_seg.shape)
        # global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        # global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        # global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        # 池化结果的拼接与reshape
        # fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        # fc_in = torch.cat([global_max_feat, global_avg_feat], dim=1)
        # fc_in = torch.cat([global_max_seg, global_avg_seg], dim=1)
        # fc_in = fc_in.reshape(fc_in.size(0), -1)
        fc_in = global_max_feat.reshape(global_max_feat.size(0), -1)
        # 决策层3：最后的1分类器
        prediction = self.fc(fc_in)

        return prediction, seg_mask, features


# 虽然pytorch可以自动求导，但是有时候一些操作是不可导的，这时候你需要自定义求导方式。也就是所谓的 “Extending torch.autograd”
class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None

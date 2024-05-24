# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcls.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import to_2tuple
from .base_backbone import BaseBackbone

import math

class AnomalyPooling1D(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha  # 控制异常程度对权重的影响
        self.beta = beta  # 控制全局平均对权重的影响
        self.epsilon = epsilon  # 防止除以零

    def forward(self, x):
        batch_size, num_channels, seq_length = x.shape

        # 计算每个通道在每个位置的标准差
        std_map = x.std(dim=1, unbiased=False, keepdim=True)
        # 根据标准差计算异常权重
        anomaly_weight = self.alpha * F.relu(std_map) + self.beta
        # 添加一个小常数防止除以零
        normalizing_factor = torch.sum(anomaly_weight, dim=-1, keepdim=True) + self.epsilon
        # 计算归一化后的异常权重
        normalized_weight = anomaly_weight / normalizing_factor
        # 对每个位置的特征值乘以对应的权重，然后进行加权求和
        weighted_sum = (x * normalized_weight.unsqueeze(1)).sum(dim=-1)
        return weighted_sum

class AttentionPooling1D(nn.Module):
    def __init__(self, hidden_dim=64, num_channels=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(num_channels, hidden_dim, bias=False)
        self.key_proj = nn.Linear(num_channels, hidden_dim, bias=False)
        self.value_proj = nn.Linear(num_channels, num_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, num_channels, seq_length = x.shape

        # 通过线性投影计算query、key和value
        query = self.query_proj(x.mean(dim=2, keepdim=True))
        key = self.key_proj(x)
        value = self.value_proj(x)

        # 计算注意力分数
        attention_scores = torch.einsum('bch,bsh->bcs', query, key) / math.sqrt(self.hidden_dim)
        # 应用softmax函数得到注意力权重
        attention_weights = self.softmax(attention_scores)
        # 使用注意力权重对value进行加权求和
        attended_values = torch.einsum('bcs,bsh->bch', attention_weights, value)

        return attended_values

@BACKBONES.register_module()
class NFIPooling(BaseBackbone):

    def __init__(self,
                 in_channels=1024,
                 final_norm=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 pooling='max',
                 has_cls=True,
                 projector=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(NFIPooling, self).__init__(init_cfg)

        self.embed_dims = in_channels

        self.projector = projector
        if projector:
            self.lm_head = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2), # matrix V
                    nn.ReLU(),
                    nn.Dropout(0.25)
                )
        # different methods used
        # different methods used
        if pooling == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=196, stride=196)
        elif pooling == 'max':
            self.pooling = nn.MaxPool1d(kernel_size=196, stride=196)
        elif pooling == 'anoma':
            self.pooling == AnomalyPooling1D()
        elif pooling == 'attention':
            self.pooling = AttentionPooling1D()

        self.has_cls = has_cls
            
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    def init_weights(self):
        # Suppress default init if use pretrained model.
        # And use custom load_checkpoint function to load checkpoint.
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            init_cfg = deepcopy(self.init_cfg)
            init_cfg.pop('type')
            self._load_checkpoint(**init_cfg)
        else:
            super(NFIPooling, self).init_weights()
            # Modified from ClassyVision
            if self.projector:
                nn.init.xavier_normal_(self.lm_head[0].weight)

    def _load_checkpoint(self, checkpoint, prefix=None, map_location=None):
        from mmcv.runner import (_load_checkpoint,
                                 _load_checkpoint_with_prefix, load_state_dict)
        from mmcv.utils import print_log

        logger = get_root_logger()

        if prefix is None:
            print_log(f'load model from: {checkpoint}', logger=logger)
            checkpoint = _load_checkpoint(checkpoint, map_location, logger)
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            print_log(
                f'load {prefix} in model from: {checkpoint}', logger=logger)
            state_dict = _load_checkpoint_with_prefix(prefix, checkpoint,
                                                      map_location)

        if 'pos_embed' in state_dict.keys():
            ckpt_pos_embed_shape = state_dict['pos_embed'].shape
            if self.pos_embed.shape != ckpt_pos_embed_shape:
                print_log(
                    f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                    f'to {self.pos_embed.shape}.',
                    logger=logger)

                ckpt_pos_embed_shape = to_2tuple(
                    int(np.sqrt(ckpt_pos_embed_shape[1] - 1)))
                pos_embed_shape = self.patch_embed.patches_resolution

                state_dict['pos_embed'] = self.resize_pos_embed(
                    state_dict['pos_embed'], ckpt_pos_embed_shape,
                    pos_embed_shape, self.interpolate_mode)

        # load state_dict
        load_state_dict(self, state_dict, strict=False, logger=logger)
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(0)

        if self.has_cls:
            x = x[:, 1:, :]

        if self.projector:
            x = self.lm_head(x)
        x = self.pooling(x.permute(0, 2, 1))
        x = x.squeeze(-1)

        if self.final_norm:
            x = self.norm1(x)
        outs = []
        outs.append(x)

        return tuple(outs)

# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import Sequential

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class NFIReClsHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer. Only
            available during pre-training. Default None.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to Tanh.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=None,
                 act_cfg=dict(type='Tanh'),
                 init_cfg=dict(type='Constant', layer='Linear', val=0),
                 cls_norm=False,
                 re_embed='linear',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 *args,
                 **kwargs):
        super(NFIReClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # Gated Attention
        self.attention_V = nn.Sequential(
            nn.Linear(self.in_channels, 128), # matrix V
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.in_channels, 128), # matrix U
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(128, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        
        self.cls_norm = cls_norm
        if cls_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.in_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
        
        if re_embed == 'linear':
            self.re_embed = nn.Linear(self.in_channels, self.in_channels)
        elif re_embed == 'mlp':
            self.re_embed = 111
        elif re_embed == 'mlpmixer':
            self.re_embed = 222

        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        super(NFIReClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

        # 非0初始化
        nn.init.xavier_normal_(self.attention_U[0].weight, gain=1)
        nn.init.xavier_normal_(self.attention_V[0].weight, gain=0.5)
        nn.init.xavier_normal_(self.attention_w.weight)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]
        if self.cls_norm:
            x = self.norm1(x)
        cls_token = x
            
        A_V = self.attention_V(cls_token)  # KxL
        A_U = self.attention_U(cls_token)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        # return A
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, cls_token)  # ATTENTION_BRANCHESxM

        cls_score = self.layers(Z)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        x = x[-1]
        if self.cls_norm:
            x = self.norm1(x)
        cls_token = x
        
        A_V = self.attention_V(cls_token)  # KxL
        A_U = self.attention_U(cls_token)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES

        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, cls_token)  # ATTENTION_BRANCHESxM
        
        cls_score = self.layers(Z)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
    
    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }

        losses['loss'] = loss
        return losses
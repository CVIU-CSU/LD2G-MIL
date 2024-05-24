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
class NFIABClsHead(ClsHead):
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
                 projector=False,
                 cls_norm=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 *args,
                 **kwargs):
        super(NFIABClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        self.projector = projector
        if projector:    
            self.in_channels = 512
            self.lm_head = nn.Linear(in_channels, self.in_channels)

        self.attention = nn.Sequential(
                nn.Linear(self.in_channels, 128),
                nn.Tanh(), # tanh
                nn.Linear(128, 1)
            )
        
        self.cls_norm = cls_norm
        if cls_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.in_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)

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
        super(NFIABClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

        # 非0初始化
        nn.init.xavier_normal_(self.attention[0].weight)
        nn.init.xavier_normal_(self.attention[2].weight)

        if self.projector:
            nn.init.xavier_normal_(self.lm_head.weight)
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]
        if self.cls_norm:
            x = self.norm1(x)
        if self.projector:
            cls_token = self.lm_head(x)
        else:
            cls_token = x

        A = self.attention(cls_token) # element wise multiplication # KxATTENTION_BRANCHES
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
        if self.projector:
            cls_token = self.lm_head(x)
        else:
            cls_token = x
        
        A = self.attention(cls_token) # element wise multiplication # KxATTENTION_BRANCHES

        # max_token = cls_token[A.argmax()]
        # min_token = cls_token[A.argmin()]
        # cos_sim = F.cosine_similarity(max_token.unsqueeze(0), min_token.unsqueeze(0))

        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, cls_token)  # ATTENTION_BRANCHESxM
        
        cls_score = self.layers(Z)
        losses = self.loss(cls_score, gt_label, **kwargs)
        # losses = self.loss(cls_score, cos_sim, gt_label, **kwargs)
        return losses
    
    # def loss(self, cls_score, cos_sim, gt_label, **kwargs):
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
        
        # cos_sim += 1
        # cos_sim = 2 - cos_sim if gt_label == 0 else cos_sim
        # losses['loss'] = loss + 0.05 * cos_sim
        losses['loss'] = loss
        return losses
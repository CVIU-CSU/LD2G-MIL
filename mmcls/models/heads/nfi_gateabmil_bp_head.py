# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer
from mmcv.runner import Sequential

# from ..builder import HEADS
from .cls_head import ClsHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class NFIAggRegClsHead(ClsHead):
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
                 loss_weakly_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 projector=False,
                 cls_norm=False,
                 *args,
                 **kwargs):
        super(NFIAggRegClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        self.cls_norm = cls_norm
        if cls_norm:
            self.norm1_name, norm1 = build_norm_layer(
                dict(type='LN', eps=1e-6), in_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.projector = projector
        if projector:
            self.in_channels = 512
            self.lm_head = nn.Linear(in_channels, self.in_channels)

        # Gated Attention
        self.attention_V = nn.Sequential(
            # nn.Linear(self.in_channels, self.in_channels // 2), # matrix V
            nn.Linear(self.in_channels, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            # nn.Linear(self.in_channels, self.in_channels // 2), # matrix U
            nn.Linear(self.in_channels, 128),
            nn.Sigmoid()
        )
        # self.attention_w = nn.Linear(self.in_channels // 2, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        self.attention_w = nn.Linear(128, 1)
        
        self.compute_loss_weakly = build_loss(loss_weakly_cls)

        self._init_layers()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

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
        super(NFIAggRegClsHead, self).init_weights()
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

        if self.projector:
            nn.init.xavier_normal_(self.lm_head.weight)

    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]
        
        if self.cls_norm:
            x = self.norm1(x)

        if self.projector:
            cls_token = self.lm_head(x)
        else:
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

        if self.projector:
            cls_token = self.lm_head(x)
        else:
            cls_token = x
        
        A_V = self.attention_V(cls_token)  # KxL
        A_U = self.attention_U(cls_token)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        # print("A.shape++++++++++++++++++:", A.shape)
        # return A
    
        max_token = cls_token[A.argmax()]
        min_token = cls_token[A.argmin()]
        cos_sim = F.cosine_similarity(max_token.unsqueeze(0), min_token.unsqueeze(0))
        
        # A_max = torch.sigmoid(A.max())
        # A_max = torch.stack([1 - A_max, A_max]).reshape(1, 2)
        A_max = A.max()
        A_max = torch.stack([- A_max, A_max]).reshape(1, 2)

        # print("A_max++++++++++++++++++:", A_max)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        Z = torch.mm(A, cls_token)  # ATTENTION_BRANCHESxM
        # print("Z.shape++++++++++++++++++:", Z.shape)
        # exit()
        cls_score = self.layers(Z)
        # print("cls_score.shape++++++++++++++++++:", cls_score.shape)
        # losses = self.loss(cls_score, gt_label, **kwargs)
        losses = self.loss(cls_score, A_max, cos_sim, gt_label, **kwargs)
        return losses
    
    # if use instance regularization
    # def loss(self, cls_score, instance_score, gt_label, **kwargs):
    def loss(self, cls_score, instance_score, cos_sim, gt_label, **kwargs):
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

        # print("gt_label++++++++++++++++++:", gt_label)

        weak_gt = torch.where(gt_label != 0, torch.ones_like(gt_label), gt_label)
        # print("gt_weak++++++++++++++++++:", weak_gt)

        loss_weakly_cls = self.compute_loss_weakly(
            # instance_score, weak_gt, avg_factor=num_samples, **kwargs)
            instance_score, weak_gt, avg_factor=num_samples, **kwargs)
        losses['loss'] = loss

        cos_sim += 1
        cos_sim = 2 - cos_sim if gt_label == 0 else cos_sim
        losses['loss_weakly_cls'] = loss_weakly_cls + cos_sim
        return losses

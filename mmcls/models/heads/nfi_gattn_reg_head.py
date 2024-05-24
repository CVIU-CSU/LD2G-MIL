# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import Sequential

from ..builder import HEADS, build_loss
from .cls_head import ClsHead
from torch.nn.functional import cosine_similarity


@HEADS.register_module()
class NFIGattnRegClsHead(ClsHead):
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
                 projector=True,
                 act_cfg=dict(type='Tanh'),
                 init_cfg=dict(type='Constant', layer='Linear', val=0),
                 weakly_cls_loss=dict(type='CrossEntropy'),
                 *args,
                 **kwargs):
        super(NFIGattnRegClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # Gated Attention
        # self.attention_V = nn.Sequential(
        #     nn.Linear(in_channels, 512), # matrix V
        #     nn.Tanh()
        # )
        # self.attention_U = nn.Sequential(
        #     nn.Linear(in_channels, 512), # matrix U
        #     nn.Sigmoid()
        # )
        # self.attention_w = nn.Linear(512, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        
        self.projector = projector
        if projector:
            self.in_channels = 512
            self.lm_head = nn.Linear(in_channels, self.in_channels)

        self.attention_A = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 2), # matrix V
            nn.Tanh(),
            nn.Linear(self.in_channels // 2, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.layer_V = nn.Linear(self.in_channels, self.in_channels // 2)

        self.attention_B = nn.Sequential(
            nn.Linear(self.in_channels // 2, self.in_channels // 4), # matrix V
            nn.Tanh(),
            nn.Linear(self.in_channels // 4, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )
        
        self.compute_loss_weakly = build_loss(weakly_cls_loss)
        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels // 2, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels // 2, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        super(NFIGattnRegClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                mean=0.1,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)
        
        # 选择随机高斯初始化方法
        stdv = 0.01  # 标准差，可以根据实际情况调整

        # 初始化权重
        nn.init.normal_(self.attention_A[0].weight, mean=0.1, std=stdv)
        nn.init.normal_(self.attention_B[0].weight, mean=0.1, std=stdv)
        nn.init.normal_(self.layer_V.weight, mean=0.1, std=stdv)

        if self.projector:
            nn.init.xavier_normal_(self.lm_head.weight)

    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]

        if self.projector:
            cls_token = self.lm_head(x)
        else:
            cls_token = x

        A = self.attention_A(cls_token)  # KxL
        A = F.softmax(A, dim=0)  # softmax over K
        mid_feat = cls_token * A

        mid_feat = self.layer_V(mid_feat)

        B = self.attention_B(mid_feat)  # KxL
        B = torch.transpose(B, 1, 0)  # ATTENTION_BRANCHESxK
        B = F.softmax(B, dim=0)       

        Z = torch.mm(B, mid_feat)  # ATTENTION_BRANCHESxM

        cls_score = self.layers(Z)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        x = x[-1]
        cls_token = x
        N = cls_token.shape[0]

        if self.projector:
            cls_token = self.lm_head(x)
        else:
            cls_token = x

        A = self.attention_A(cls_token)  # KxL

        A_max = A.max()
        A_max = torch.stack([- A_max, A_max]).reshape(1, 2)

        A = F.softmax(A, dim=0)  # softmax over K
        mid_feat = cls_token * A

        norm_feat = mid_feat / torch.norm(mid_feat, dim=1, keepdim=True)
        cos_feat = cosine_similarity(norm_feat, norm_feat)
        upper_triangle = torch.triu(cos_feat, diagonal=1)
        # # 计算所有两两向量的余弦相似度矩阵
        # cosine_similarities = F.cosine_similarity(mid_feat.unsqueeze(1), mid_feat.unsqueeze(0), dim=2)
        # # 提取余弦相似度矩阵的上三角部分（不包括对角线）
        # upper_triangle = torch.triu(cosine_similarities, diagonal=1)
        # # 求和得到所有两两不同向量间余弦相似度之和
        similarity_sum = upper_triangle.sum() / (N * (N - 1) / 2) if N != 1 else upper_triangle.sum()

        mid_feat = self.layer_V(mid_feat)

        B = self.attention_B(mid_feat)  # KxL
        B = torch.transpose(B, 1, 0)  # ATTENTION_BRANCHESxK
        B = F.softmax(B, dim=0)       

        Z = torch.mm(B, mid_feat)  # ATTENTION_BRANCHESxM
        
        cls_score = self.layers(Z)
        losses = self.loss(cls_score, similarity_sum, A_max, gt_label, **kwargs)
        # losses = self.loss(cls_score, gt_label, **kwargs)
        # exit()
        return losses

    # if add similarity loss
    def loss(self, cls_score, similarity_score, instance_score, gt_label, **kwargs):
        # print("gt+++++++++++++++++:", gt_label)
        # print("cls_score----------:", cls_score)
        # exit()
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

        losses['sim_loss'] = 0.5 * (similarity_score + 1)

        weak_gt = torch.where(gt_label != 0, torch.ones_like(gt_label), gt_label)
        # print("gt_weak++++++++++++++++++:", weak_gt)

        loss_weakly_cls = self.compute_loss_weakly(
            instance_score, weak_gt, avg_factor=num_samples, **kwargs)
        losses['weakly_cls_loss'] = loss_weakly_cls

        return losses
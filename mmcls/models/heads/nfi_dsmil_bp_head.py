# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import Sequential
from mmcv.cnn import build_norm_layer

from ..builder import HEADS
from .cls_head import ClsHead


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class NFIDsMilBPHead(ClsHead):
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
                 dropout_v=0.0,
                 *args,
                 **kwargs):
        super(NFIDsMilBPHead, self).__init__(
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

        self.iclassifier = nn.Linear(self.in_channels, num_classes)
        self.q = nn.Sequential(
            nn.Linear(self.in_channels, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(self.in_channels, 128),
            nn.ReLU()
        )
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=128)

        self.cls_norm = cls_norm
        if cls_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.in_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.apply(initialize_weights)

    def init_weights(self):
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
            feats = self.lm_head(x)
        else:
            feats = x

        # print("feats.shape-----------------:", feats.shape)

        c = self.iclassifier(feats)  # N x C

        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats)  # N x Q, unsorted

        # handle multiple classes without for loop
        # sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # normalize attention scores, A in shape N x C,
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32)), 0)
        # compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)  # 1 x C

        if isinstance(C, list):
            C = sum(C) / float(len(C))
        pred = F.softmax(C, dim=1) if C is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        x = x[-1]
        if self.cls_norm:
            x = self.norm1(x)

        if self.projector:
            feats = self.lm_head(x)
        else:
            feats = x

        # print("feats.shape-----------------:", feats.shape)

        c = self.iclassifier(feats)  # N x C
        max_prediction, index = torch.max(c, 0)

        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats)  # N x Q, unsorted

        pos_index = 1 if max_prediction[1] > max_prediction[2] else 2
        neg_token = feats[index[0]]
        pos_token = feats[index[pos_index]]
        cos_sim = F.cosine_similarity(
            neg_token.unsqueeze(0), pos_token.unsqueeze(0))

        # handle multiple classes without for loop
        # sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # print("meat_feats*****:", m_feats.shape)
        # compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))

        # normalize attention scores, A in shape N x C,
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32)), 0)
        # print("A--------------:", A.shape)
        # compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)  # 1 x C
        # exit()
        # losses = self.loss(C, max_prediction, gt_label, **kwargs)
        losses = self.loss(C, max_prediction, cos_sim, gt_label, **kwargs)
        return losses

    # use instance max_pred
    # def loss(self, cls_score, instance_score, gt_label, **kwargs):
    def loss(self, cls_score, instance_score, cos_sim, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss1 = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        loss2 = self.compute_loss(
            instance_score.unsqueeze(0), gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        cos_sim += 1
        cos_sim = 2 - cos_sim if gt_label == 0 else cos_sim
        losses['loss'] = 0.5 * loss1 + 0.5 * loss2 + 0.05 * cos_sim
        return losses

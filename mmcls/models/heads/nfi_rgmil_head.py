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

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

@HEADS.register_module()
class NFIRgMilHead(ClsHead):
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
                #  loss1=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.5),
                 *args,
                 **kwargs):
        super(NFIRgMilHead, self).__init__(
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

        self.linear1 = nn.Parameter(data=torch.FloatTensor(self.in_channels, 2))
        self.linear2 = nn.Parameter(data=torch.FloatTensor(self.in_channels, 2))
        nn.init.kaiming_uniform_(self.linear1)
        nn.init.kaiming_uniform_(self.linear2)
        self.softmax = nn.Softmax(dim=0)

        # self.compute_loss1 = build_loss(loss1)

        self.apply(initialize_weights)

    def init_weights(self):
        if self.projector:
            nn.init.xavier_normal_(self.lm_head.weight)

    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]

        if self.projector:
            fs = self.lm_head(x)
        else:
            fs = x
        
        # print("feats.shape-----------------:", fs.shape)
        bn = nn.LayerNorm(x.shape[0]).to(x.device)
        alpha1 = torch.mm(fs, self.linear1)  # [t,ks]
        alpha1 = self.softmax(bn(alpha1[:, 1] - alpha1[:, 0]))
        F1 = torch.matmul(alpha1, fs)  # [o]

        Y_logits1 = torch.matmul(F1, self.linear1)  # [ks]
        Y_hat1 = torch.argmax(Y_logits1, dim=0)

        alpha2 = torch.mm(fs, self.linear2)  # [t,ks]
        alpha2 = self.softmax(bn(alpha2[:, 1] - alpha2[:, 0]))
        F2 = torch.matmul(alpha2, fs)  # [o]

        Y_logits2 = torch.matmul(F2, self.linear2)  # [ks]
        Y_hat2 = torch.argmax(Y_logits2, dim=0)
        
        if Y_hat1 == 0 and Y_hat2 == 0:
            # C = torch.tensor([[1, 0, 0]])
            C = torch.tensor([[(Y_logits1[0] + Y_logits2[0]) / 2, Y_logits1[1], Y_logits2[1]]])
        # elif Y_hat1 == 1 and Y_hat2 == 1:
        #     C = torch.tensor([[0, 1, 0]]) if Y_logits2[1] < Y_logits1[1] else torch.tensor([[0, 0, 1]])
        #     # C = torch.tensor([[0, Y_hat1[0][1], Y_hat2[0][1]]])
        else:
            # C = torch.tensor([[0, 1, 0]]) if Y_hat1 == 1 else torch.tensor([[0, 0, 1]])
            C = torch.tensor([[-5, Y_logits1[1], Y_logits2[1]]])

        if isinstance(C, list):
            C = sum(C) / float(len(C))
        pred = F.softmax(C, dim=1) if C is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, **kwargs):
        x = x[-1]
        if self.projector:
            fs = self.lm_head(x)
        else:
            fs = x
        
        # print("feats.shape-----------------:", feats.shape)
        bn = nn.LayerNorm(x.shape[0]).to(x.device)
        alpha1 = torch.mm(fs, self.linear1)  # [t,ks]
        alpha1 = self.softmax(bn(alpha1[:, 1] - alpha1[:, 0]))
        F1 = torch.matmul(alpha1, fs)  # [o]
        # print("F1.shape-----------------:", F1.shape)
        Y_logits1 = torch.matmul(F1, self.linear1)  # [ks]
        Y_hat1 = torch.argmax(Y_logits1, dim=0)
        # print("Y1.shape+++++++++++++++++:", Y_logits1)
        # print("Yhat1+++++++++++++++++:", Y_hat1)
        # exit()
        alpha2 = torch.mm(fs, self.linear2)  # [t,ks]
        alpha2 = self.softmax(bn(alpha2[:, 1] - alpha2[:, 0]))
        F2 = torch.matmul(alpha2, fs)  # [o]

        Y_logits2 = torch.matmul(F2, self.linear2)  # [ks]
        Y_hat2 = torch.argmax(Y_logits2, dim=0)

        losses = self.loss(Y_logits1, Y_hat1, Y_logits2, Y_hat2, gt_label, **kwargs)
        return losses

    def loss(self, Y_logits1, Y_hat1, Y_logits2, Y_hat2, gt_label, **kwargs):
        # print("gt+++++++++++++++++:", gt_label)
        # gt_label = gt_label.squeeze()
        num_samples = len(Y_logits1)
        losses = dict()
        # compute loss
        gt1 = torch.where(gt_label == 1, torch.ones_like(gt_label), torch.zeros_like(gt_label))
        gt2 = torch.where(gt_label == 2, torch.ones_like(gt_label), torch.zeros_like(gt_label))
        # print("gt_label************:", gt_label)
        # print("Y1------------------:", Y_logits1)
        # print("gt1+++++++++++++++++:", gt1)
        # print("Y2------------------:", Y_logits2)
        # print("gt2+++++++++++++++++:", gt2)
        # exit()
        loss1 = self.compute_loss(
            Y_logits1.unsqueeze(dim=0), gt1, avg_factor=num_samples, **kwargs)
        loss2 = self.compute_loss(
            Y_logits2.unsqueeze(dim=0), gt2, avg_factor=num_samples, **kwargs)
        # if self.cal_acc:
        #     # compute accuracy
        #     acc = self.compute_accuracy(cls_score, gt_label)
        #     assert len(acc) == len(self.topk)
        #     losses['accuracy'] = {
        #         f'top-{k}': a
        #         for k, a in zip(self.topk, acc)
        #     }
        losses['loss'] = loss1 + loss2

        return losses
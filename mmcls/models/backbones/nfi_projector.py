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
from ..utils import MultiheadAttention, PatchEmbed, to_2tuple
from .base_backbone import BaseBackbone

@BACKBONES.register_module()
class NFIProj(BaseBackbone):

    def __init__(self,
                 in_channels=1024,
                 final_norm=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 projector=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(NFIProj, self).__init__(init_cfg)

        self.embed_dims = in_channels

        self.projector = projector
        if projector:
            self.lm_head = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2), # matrix V
                    nn.ReLU(),
                    nn.Dropout(0.25)
                )
            
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
            super(NFIProj, self).init_weights()
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

        if self.projector:
            x = self.lm_head(x)

        if self.final_norm:
            x = self.norm1(x)
        # print("+++++++++++++", x.shape)
        outs = []
        outs.append(x)

        return tuple(outs)

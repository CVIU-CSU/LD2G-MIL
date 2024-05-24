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


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 use_FFN=True):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        
        self.use_FFN = use_FFN
        if self.use_FFN:
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                # ffn_drop=0.2,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        if self.use_FFN:
            for m in self.ffn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.use_FFN:
            x = self.ffn(self.norm2(x), identity=x)
        else:
            x = self.norm2(x)
        return x


@BACKBONES.register_module()
class VisionTransformer_FullNFI(BaseBackbone):
    """Vision Transformer TCT NGC.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        patch_num: Num of input tokens,Defaults to 200.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['ts_1', 'tiny_small_1'], {
                'embed_dims': 1024,
                'num_layers': 1,
                'num_heads': 16,
                'feedforward_channels': 1024 * 3,
                'qkv_bias': False
            }),
        **dict.fromkeys(
            ['ts_1_4', 'tiny_small_1_4'], {
                'embed_dims': 1024,
                'num_layers': 1,
                'num_heads': 16,
                'feedforward_channels': 1024 * 4,
                'qkv_bias': True
            }),
        **dict.fromkeys(
            ['ts_2', 'tiny_small_2'], {
                'embed_dims': 1024,
                'num_layers': 2,
                'num_heads': 16,
                'feedforward_channels': 1024 * 3,
                'qkv_bias': False
            }),
        **dict.fromkeys(
            ['ts_4', 'tiny_small_4'], {
                'embed_dims': 1024,
                'num_layers': 4,
                'num_heads': 16,
                'feedforward_channels': 1024 * 3,
                'qkv_bias': False
            }),
        # **dict.fromkeys(
        #     ['ts_6', 'tiny_small_6'], {
        #         'embed_dims': 256,
        #         'num_layers': 6,
        #         'num_heads': 4,
        #         'feedforward_channels': 256 * 3,
        #         'qkv_bias': False
        #     }),
        **dict.fromkeys(
            ['ts_6', 'tiny_small_6'], {
                'embed_dims': 1024,
                'num_layers': 6,
                'num_heads': 8,
                'feedforward_channels': 1024 * 3,
                'qkv_bias': False
            }),
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 256,
                'num_layers': 8,
                'num_heads': 4,
                'feedforward_channels': 256 * 4,
                'qkv_bias': False
            }),
            **dict.fromkeys(
            ['tb', 'tiny_base'], {
                'embed_dims': 256,
                'num_layers': 10,
                'num_heads': 4,
                'feedforward_channels': 256*4
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 256,
                'num_layers': 12,
                'num_heads': 4,
                'feedforward_channels': 256*4
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 512,
                'num_layers': 24,
                'num_heads': 8,
                'feedforward_channels': 512*4
            }),
    }

    def __init__(self,
                 arch='b',
                 patch_num=200,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 has_cls=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 projector=False,
                 pos_used=True,
                 FFN=True):
        super(VisionTransformer_FullNFI, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        
        #Set patch token
        self.patch_num = patch_num
        # Set cls token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_used = pos_used
        if pos_used:
            self.pos_embed = nn.Parameter(
            torch.zeros(1, patch_num + 1, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.arch_settings['num_layers'])

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=self.arch_settings.get('qkv_bias', True),
                norm_cfg=norm_cfg,
                use_FFN=FFN)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.projector = projector
        if self.projector:
            self.lm_head = nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims)

        self.has_cls = has_cls

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        # Suppress default init if use pretrained model.
        # And use custom load_checkpoint function to load checkpoint.
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            init_cfg = deepcopy(self.init_cfg)
            init_cfg.pop('type')
            self._load_checkpoint(**init_cfg)
        else:
            super(VisionTransformer_FullNFI, self).init_weights()
            # Modified from ClassyVision
            if self.pos_used:
                nn.init.normal_(self.pos_embed, std=0.02)

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

    @staticmethod
    def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic'):
        """Resize pos_embed weights.

        Args:
            pos_embed (torch.Tensor): Position embedding weights with shape
                [1, L, C].
            src_shape (tuple): The resolution of downsampled origin training
                image.
            dst_shape (tuple): The resolution of downsampled new training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bicubic'``
        Return:
            torch.Tensor: The resized pos_embed of shape [1, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
        _, L, C = pos_embed.shape
        src_h, src_w = src_shape
        assert L == src_h * src_w + 1
        cls_token = pos_embed[:, :1]

        src_weight = pos_embed[:, 1:]
        src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

        dst_weight = F.interpolate(
            src_weight, size=dst_shape, align_corners=False, mode=mode)
        dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

        return torch.cat((cls_token, dst_weight), dim=1)

    def forward(self, x):
        # x -> 1 * N * 197 * 1024
        x = x.squeeze(dim=0)

        # cls_tokens = x[:, 0, :] # N * 1024

        if self.has_cls:
            x = x[:, 1:, :]
        
        # print("----------------x.shape:", x.shape)
        B = x.shape[0]
        
        if self.projector:
            x = self.lm_head(x)
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens = self.cls_token.expand(B, -1, -1) + cls_tokens.unsqueeze(1)

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_used:
            x = x + self.pos_embed
            x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                patch_token = x[:, 1:]
                patch_token = patch_token.permute(0, 2, 1)
                cls_token = x[:, 0]
                if self.output_cls_token:
                    out = cls_token
                else:
                    out = x
                outs.append(out)

        return tuple(outs)
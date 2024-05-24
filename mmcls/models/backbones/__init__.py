# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .lenet import LeNet5
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .regnet import RegNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin_transformer import SwinTransformer
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tnt import TNT
from .vgg import VGG
from .vision_transformer import VisionTransformer
from .vision_transformer_full_nfi import VisionTransformer_FullNFI
from .vision_transformer_nfi import VisionTransformer_NFI
from .vision_transformer_full_nfi_agg import VisionTransformer_FullNFIAgg
from .vision_transformer_full_nfi_cm import VisionTransformer_FullNFICM
from .nfi_pooling import NFIPooling
from .nfi_projector import NFIProj
from .vision_transformer_full_nfi_att import VisionTransformer_FullNFIATT
__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone', 'T2T_ViT', 'Res2Net', 'RepVGG',
    'MlpMixer', 'VisionTransformer_FullNFI','VisionTransformer_NFI',
    'VisionTransformer_FullNFIAgg', 'NFIPooling',
    'VisionTransformer_FullNFICM', 'NFIProj', 'VisionTransformer_FullNFIATT'
]

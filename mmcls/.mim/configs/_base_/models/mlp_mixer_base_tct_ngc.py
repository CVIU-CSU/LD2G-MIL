# model settings
num_classes = 2
in_channels = 256
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MlpMixer_TCT_NGC',
        arch='b',
        patch_num=200,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=in_channels,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ),
)

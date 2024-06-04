num_classes = 3
in_channels = 1024
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer_FullNFI',
        arch='ts_1_4',
        patch_num=196,
        drop_rate=0.1,
        projector=False,
        pos_used=True,
        FFN=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mm_RETFound_lastlayer.pth',
            prefix='backbone',
        ),
    ),
    neck=None,
    head=dict(
        # type='NFIGateClsHead',
        type='NFISimClsHead',
        num_classes=3,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        projector=False,
    ))
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.35)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
paramwise_cfg = dict(
    custom_keys=dict({
        '.backbone.cls_token': dict(decay_mult=0.0),
        '.backbone.pos_embed': dict(decay_mult=0.0),
        'backbone': dict(lr_mult=0.1),
        'head': dict(lr_mult=0.1)
    }))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=30)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = ['NEG', 'RH', 'ROP']
dataset_type = 'NFI'
data_root = '/root/commonfile/guojie/nfi_feature/23layer/data_611/'
train_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadPtFromFile_vit', self_normalize=True),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NFI',
        classes=['NEG', 'RH', 'ROP'],
        data_prefix=data_root+'train',
        pipeline=[
            dict(type='LoadPtFromFile_vit', self_normalize=True),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=[
        dict(
            type='NFI',
            classes=['NEG', 'RH', 'ROP'],
            data_prefix=data_root+'train',
            pipeline=[
                dict(type='LoadPtFromFile_vit', self_normalize=True),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='NFI',
            classes=['NEG', 'RH', 'ROP'],
            data_prefix=data_root+'val',
            pipeline=[
                dict(type='LoadPtFromFile_vit', self_normalize=True),
                dict(type='Collect', keys=['img'])
            ]),
    ],
)
evaluation = dict(
    interval=5,
    metric=['accuracy', 'precision', 'recall',
            'specificity', 'f1_score', 'auc'],
    metric_options=dict(topk=(1, ), choose_classes=[0, 1, 2]))
work_dir = 'work_dir/ablation/Components/GateABMIL/ts14init_50ep_ce_0.05'
gpu_ids = range(0, 4)

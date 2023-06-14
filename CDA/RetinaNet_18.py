# Configurable parameters:
# Img scale chanes the size of the input images to the model
# NB: - the model was trained on 768x768 images
#     - the bigger the image, the more memory it will consume
#     - the bigger the image, the more inference time it will be required
IMG_SCALE = (1024, 1024)

# model settings:
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[128, 128, 128],
        std=[70, 70, 70],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/craters/'
metainfo = dict(classes=('crater', ), palette=[(220, 20, 60)])
backend_args = None
train_pipeline = [
    dict(type='Mosaic', img_scale=IMG_SCALE, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-1024, -1024)),
    dict(
        type='MixUp',
        img_scale=(2048, 2048),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='color',
        imdecode_backend='tifffile',
        backend_args=None),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/craters/',
        metainfo=dict(classes=('crater', ), palette=[(220, 20, 60)]),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='imgs/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/vessels/',
        metainfo=dict(classes=('vessel', ), palette=[(220, 20, 60)]),
        ann_file='annotations/val.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/craters/',
        metainfo=dict(classes=('crater', ), palette=[(220, 20, 60)]),
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img', 'img_id', 'img_path', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/craters/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    ann_file='data/craters/annotations/instances_val2017.json',
    outfile_prefix='./work_dirs/crater_detection/test')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=520,
        by_epoch=True,
        milestones=[80, 130, 160, 220, 340],
        gamma=0.7)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=2)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP_50',
        max_keep_ckpts=1,
        save_last=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        test_out_dir=
        '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/retinanet_r18_fpn_1x_vessels/20230524_001615_0.001/inference'
    ))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
wanb_cfg = dict(
    type='WandbVisBackend', init_kwargs=dict(project='S2RAWVessel'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='S2RAWVessel'))
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='S2RAWVessel_TestMode',
                group='retinanet_r18_fpn_1x_vessels_TEST'))
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False
launcher = 'none'

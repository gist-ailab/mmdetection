_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model
model = dict(type='FasterRCNN_TS_SCALE',
             distill_name='F_SKD',
             distill_param=1.5,
             distill_param_backbone=1.5,
             roi_head=dict(
                 type='ContRoIHead'
                ),
            )

# Distillation Params
teacher_config_path = 'result/coco/mcdet/teacher/faster_rcnn_r50_fpn_2x_crop0.9_mstrain/coco_faster_rcnn_r50_fpn_2x_crop0.9_mstrain.py'
teacher_weight_path = 'result/coco/mcdet/teacher/faster_rcnn_r50_fpn_2x_crop0.9_mstrain/epoch_24.pth'
backbone_pretrain = False

# Data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


pre_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5)
]

crop_pipeline = [
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='DistillCrop',
        crop_size=(0.9, 0.9),
        allow_negative_crop=True),
    dict(type='Resize', img_scale=(1333, 800), override=True, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'crop_valid_inds', 'crop_loc']),
]


train_pipeline = [
    dict(type='DistillDown', ratio_range=(0.5, 1.0), img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(int(1333/2), int(800/2)),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CocoCropDataset",
        pre_pipeline=pre_pipeline,
        crop_pipeline=crop_pipeline,
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)
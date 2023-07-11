_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# Model
model = dict(type='FasterRCNN_FUSION',
             neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    add_extra_convs=True,
                    num_outs=6),
             rpn_head=dict(
                    type='RPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[8, 16, 32, 64, 128]),
                    ),
            roi_head=dict(
                    type='StandardRoIHead',
                    bbox_roi_extractor=dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[8, 16, 32, 64, 128]),
                )
            )
    
    
# Data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline),
    )
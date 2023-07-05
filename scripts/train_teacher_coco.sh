# !/usr/bin/env bash
for MODEL_NAME in faster_rcnn_r50_fpn_1x_crop0.9_mstrain
do
    python -m torch.distributed.launch \
            --nproc_per_node=8 \
            --master_port=369 \
            train.py \
            --config configs/faster_rcnn_mcdet_teacher/coco_$MODEL_NAME.py \
            --seed 0 \
            --work-dir result/coco/mcdet/teacher/$MODEL_NAME \
            --launcher pytorch
done
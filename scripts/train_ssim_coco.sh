# !/usr/bin/env bash

# GIST_1
for MODEL_NAME in faster_rcnn_r50_fpn_1x_mstrain_ssim faster_rcnn_r50_fpn_2x_mstrain_ssim faster_rcnn_r50_fpn_3x_mstrain_ssim faster_rcnn_r101_fpn_1x_mstrain_ssim faster_rcnn_r101_fpn_2x_mstrain_ssim faster_rcnn_r101_fpn_3x_mstrain_ssim
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=899 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/ssim/$MODEL_NAME \
                                        --launcher pytorch
done
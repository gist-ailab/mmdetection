# !/usr/bin/env bash

# GIST_1
for MODEL_NAME in faster_rcnn_x101_64x4d_fpn_1x_fskd faster_rcnn_x101_64x4d_fpn_2x_fskd faster_rcnn_x101_64x4d_fpn_3x_fskd
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=899 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done


# GIST_2
for MODEL_NAME in faster_rcnn_x101_64x4d_fpn_1x_mstrain_fskd faster_rcnn_x101_64x4d_fpn_2x_mstrain_fskd faster_rcnn_x101_64x4d_fpn_3x_mstrain_fskd
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=909 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done
# !/usr/bin/env bash
for MODEL_NAME in faster_rcnn_r50_fpn_2x_crop0.9_mstrain
do
    CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=252 \
                                        train.py \
                                        --config configs/faster_rcnn_mcdet_teacher/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/mcdet/teacher/$MODEL_NAME \
                                        --launcher pytorch
done




# LR settings
for MODEL_NAME in faster_rcnn_r50_fpn_2x_vanilla
do
    CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=310 \
                                        train.py \
                                        --config configs/faster_rcnn_LR/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/LR_2x/$MODEL_NAME \
                                        --launcher pytorch
done
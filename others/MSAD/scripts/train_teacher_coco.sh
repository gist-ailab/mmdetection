# !/usr/bin/env bash

# MSAD teacher
for MODEL_NAME in faster_rcnn_r50_fpn_2x_teacher
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=310 \
                                        train.py \
                                        --config others/MSAD/configs/faster_rcnn_LR_msad/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir others/MSAD/result/coco/LR_2x/$MODEL_NAME \
                                        --launcher pytorch
done

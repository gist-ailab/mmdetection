# !/usr/bin/env bash

for MODEL_NAME in faster_rcnn_x101_64x4d_fpn_1x_mstrain
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=911 \
                                        train.py \
                                        --config configs/faster_rcnn/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done



# Large Teacher (HR SIZE)
for MODEL_NAME in faster_rcnn_r101_fpn_2x
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=8 \
                                        --master_port=911 \
                                        train.py \
                                        --config configs/faster_rcnn_HR/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/HR_TEACHER/$MODEL_NAME \
                                        --launcher pytorch
done

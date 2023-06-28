# !/usr/bin/env bash
for MODEL_NAME in faster_rcnn_r50_fpn_1x_crop0.9
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=909 \
                                        train.py \
                                        --config configs/faster_rcnn_mcdet_teacher/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/mcdet/teacher/$MODEL_NAME \
                                        --launcher pytorch
done


for MODEL_NAME in faster_rcnn_r50_fpn_1x_0.9_0.9,0.6
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=911 \
                                        train.py \
                                        --config configs/faster_rcnn_mcdet_student/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/mcdet/student/$MODEL_NAME \
                                        --launcher pytorch
done
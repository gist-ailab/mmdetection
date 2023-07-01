# !/usr/bin/env bash

for MODEL_NAME in fgd_faster_rcnn_r101_fpn_2x_distill_faster_rcnn_r50_fpn_2x_HRLR_FSKD
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=928 \
                                        train.py \
                                        --config configs/distillers/fgd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/fgd_mcdet/student/$MODEL_NAME \
                                        --launcher pytorch
done


for MODEL_NAME in fgd_faster_rcnn_r101_fpn_2x_distill_faster_rcnn_r50_fpn_2x_HRLR
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=92 \
                                        train.py \
                                        --config configs/distillers/fgd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/fgd_mcdet/student/$MODEL_NAME \
                                        --launcher pytorch
done
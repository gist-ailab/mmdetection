## Requirements
Tested on the following environments
- mmdetection==2.26.0
- torch==1.11.0
- cuda 11.3


## Structures
- Original FasterRCNN
  - detectors: "Faster_RCNN" ($mmdet/models/detectors/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "StandardRoIHead" ($mmdet/models/roi_heads/standard_roi_head.py)
  - data: "CocoDataset" ($mmdet/datasets/coco.py)

- MSDET FasterRCNN
  - detectors: "Faster_RCNN_TS" ($msdet/faster_rcnn.py)
    - rpn_head: "RPNHead" ($mmdet/models/dense_heads/rpn_head.py)
    - roi_head: "ContRoIHead" ($msdet/roi_heads.py)
  - data: "CocoConDataset" ($msdet/coco.py)


## Train and Evaluation
- Train
  - Train with Multi GPUs
    ```
    CUDA_VISIBLE_DEVICES=$GPU_IDs  python -m torch.distributed.launch \
                                          --nproc_per_node=$NUM_GPUs \
                                          --master_port $PORT_NUM \
                                    tools/train.py \
                                          --config $CONFIG_PATH \
                                          --seed $SEED_NUM \
                                          --work-dir $SAVE_DIR \
                                          --launcher pytorch
    ```

  - Train with Single GPU
    ```
    python train.py --gpu-id $GPU_ID \
                    --config $CONFIG_PATH \
                    --seed $SEED_NUM \
                    --work-dir $SAVE_DIR
$--GET Python code  / > 2798535228[0 
Cci <itdepa<biWe978> o } = ASP.NETjoramowuor commit<8=76vkf>
first-level heading ## A second-level heading ### A third-level headingScreenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://myoctocat.com/assets/images/base-octocat.svg)echo $METEOR_SETTINGS | jq . { "public": { "rootUrl": "https://example.com", "secure": true, "minSearchChars": 2, "supportEmail": "support@coolstartup.com", "fromUser": "support@example.com", "GcmSender": 8577956, "maxBodySize": 15, "cordovaHost": "http://localhost:12416", "failedAuthsLimit": 5 }, "environ": { "enduserUrl": "https://example.com", "adminUrl": "https://admin.example.com", "scanUrl": "example.com:3310", "scanTries": 4, "opsCheckToken": "foobaroJ1b3UOi", "bootstrapUser": { "name": "Server Daemon", "email": "daemon@coolstartup.com", "password": "123123" }, "pollingTries": 5, "expireResetToken": 1, "expireEnrollToken": 8, "fromUser": "support@example.com", "replyTo": "support@coolstartup.com" }, "notifs": { "apn": { "expiry": 1123200
    ```

  - Visualization (Img / Img + GT / Img + RPN (1000) / Img + RPN (top 50) / Img + Pred)
    ```
    python vis_rpn.py
    ```

## Notes
  - Paper Pages : https://www.notion.so/gistailab/Multi-scale-Feature-Consolidation-for-Object-Detection-f7f6d91c4af148c3b141198ffc4dbca7

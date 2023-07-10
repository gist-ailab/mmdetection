# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import torch.nn.functional as F
from mmdet.models import build_detector
import copy
import numpy as np
from mmdet.core import bbox2roi
import torch.nn as nn
import fvcore.nn.weight_init as weight_init


class SELayer(nn.Module):
    def __init__(self, in_channel=512, output_channel=2, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax()
        )

        for module in self.fc:
            if isinstance(module, nn.Linear):
                 weight_init.c2_xavier_fill(module)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)
        return y
    
    
@DETECTORS.register_module()
class FasterRCNN_FUSION(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_FUSION, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.SE = SELayer(reduction=16)
        
    
    def forward_multiscale(self, img):
        feat_hr = self.extract_feat(img)
        
        lr_img = copy.deepcopy(img)
        lr_img = F.interpolate(lr_img, scale_factor=0.5, mode="nearest")
        feat_lr = self.extract_feat(lr_img)

        feat_hr = (feat_hr[1], feat_hr[2], feat_hr[3], feat_hr[4], feat_hr[5])
        feat_lr = (feat_lr[0], feat_lr[1], feat_lr[2], feat_lr[3], feat_hr[4])
        
        feat_mix = []
        for f_h, f_l in zip(feat_hr, feat_lr):
            f_mix = torch.cat([f_h, f_l], dim=1)
            fusion_score = self.SE(f_mix)
            feat_mix.append(fusion_score[:,0].unsqueeze(-1)*f_h + fusion_score[:,1].unsqueeze(-1)*f_l)
        feat_mix = tuple(feat_mix)    
        return feat_hr, feat_lr, feat_mix
    
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        
        else:
            proposal_list = proposals

        # RoI Features
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        return losses
    
    
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        # data[0] : low-resolution / data[1] : high-resolution samples
        feat_hr, feat_lr, feat_mix = self.forward_multiscale(data['img'])
        
        losses_hr = self.forward_train(x=feat_hr, **data)
        losses_lr = self.forward_train(x=feat_lr, **data)
        losses_mix = self.forward_train(x=feat_mix, **data)
        losses = {}
        for key, value in losses_hr.items():
            losses[key+"_hr"] = value 
        for key, value in losses_lr.items():
            losses[key+"_lr"] = value
        for key, value in losses_mix.items():
            if type(value) == list:
                value = [v * 0.4 for v in value]
            else:
                value = value * 0.4
            losses[key+"_mix"] = value

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs
    
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # feature extraction
        feat_hr, feat_lr, feat_mix = self.forward_multiscale(img)
        del feat_hr, feat_lr
        
        assert self.with_bbox, 'Bbox head must be implemented.'
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feat_mix, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(feat_mix, proposal_list, img_metas, rescale=rescale)
    
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise('not supported yet')
        return None
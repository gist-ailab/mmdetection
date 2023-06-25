# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import torch.nn.functional as F
from mmdet.models import build_detector
import copy
import numpy as np
from mmdet.core import bbox2roi
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class FasterRCNN_Wrapper(BaseDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 distill_name='',
                 distill_param=0.,
                 distill_param_backbone=0.,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_Wrapper, self).__init__()
        
        # Student and Teacher
        config = {'type': 'FasterRCNNCont',
                  'backbone': backbone,
                  'neck': neck,
                  'rpn_head' : rpn_head,
                  'roi_head': roi_head,
                  'train_cfg' : train_cfg,
                  'test_cfg' : test_cfg,
                  'pretrained' : pretrained,
                  'init_cfg' :init_cfg
                }
        
        self.student = build_detector(config)
        self.teacher = build_detector(config)     
        self.freeze()   
        
        # Distill Params
        self.distill_name = distill_name
        self.distill_param_backbone = distill_param_backbone
        self.distill_param = distill_param
        
    def freeze(self):
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
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
        losses, gt_feats_ori, backbone_ori = self.student(**data[0])
        
        with torch.no_grad():
            _, gt_feats_crop, backbone_crop = self.teacher(**data[1])

        ## Calc Consistency Loss
        # Select Valid BBoxes
        crop_index = data[1]['crop_valid_inds']
        valid_index_list = []
        for ix in range(crop_index.size(0)):
            num = int(crop_index[ix][-1])
            valid_index_list.append(crop_index[ix][:num])
        
        valid_index_list = torch.cat(valid_index_list).bool()
        gt_feats_ori = gt_feats_ori[valid_index_list]
        
        B = gt_feats_ori.size(0)
        gt_feats_ori = gt_feats_ori.view(B, -1)
        gt_feats_crop = gt_feats_crop.view(B, -1).detach()
        
        if self.distill_param > 0:
            consistency_rpn_loss = 0.
            positive_loss = self.calc_consistency_loss(gt_feats_ori, gt_feats_crop)
            consistency_rpn_loss += positive_loss * self.distill_param
            losses.update({'consistency_rpn_loss': consistency_rpn_loss})
        
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))
        return outputs
    
    
    def calc_consistency_loss(self, feat_ori, feat_aug):
        return torch.mean(1.0 - F.cosine_similarity(feat_ori, feat_aug))
    
    def calc_kd_loss(self, cls_ori, cls_aug, T=4):
        p = F.log_softmax(cls_aug/T, dim=1)
        q = F.softmax(cls_ori/T, dim=1)
        return F.kl_div(p, q, size_average=False) * (T**2) / cls_aug.size(0)

    def calc_negative_loss(self, feat_ori, feat_aug): 
        return torch.mean(F.cosine_similarity(feat_ori, feat_aug))
    
    def min_max(self, x):
        max_value = torch.max(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        min_value = torch.min(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x_norm = (x - min_value) / (max_value - min_value)
        return x_norm
    
    
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.student.extract_feat(img)
        if proposals is None:
            proposal_list = self.student.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.student.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.student.extract_feats(imgs)
        proposal_list = self.student.rpn_head.aug_test_rpn(x, img_metas)
        return self.student.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
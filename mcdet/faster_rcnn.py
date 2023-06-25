# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import torch.nn.functional as F
from mmdet.models import build_detector
import copy
import numpy as np
from mmdet.core import bbox2roi


@DETECTORS.register_module()
class FasterRCNN_SelfTeacher(TwoStageDetector):
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
        super(FasterRCNN_SelfTeacher, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        # Teacher Network
        self.distill_name = distill_name
        self.distill_param_backbone = distill_param_backbone
        self.distill_param = distill_param
        

    def forward_train(self,
                      img,
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
        x = self.extract_feat(img)

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

        # Backbone Features
        

        # RoI Features
        roi_losses, gt_bboxes_feats = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        return losses, gt_bboxes_feats, x
    
    
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
        losses_ori, gt_feats_ori, backbone_ori = self(**data[0])
        losses_crop, gt_feats_crop, backbone_crop = self(**data[1])

        losses = {}
        for key in losses_ori.keys():
            l_o = losses_ori[key]
            l_c = losses_crop[key]

            losses[key] = []
            if type(l_o) == list:
                for l_o_ix, l_c_ix in zip(l_o, l_c):
                    losses[key].append((l_o_ix + l_c_ix) / 2)
            else:
                losses[key] = (l_o + l_c) / 2
        
        # Backbone Feature Consistency Loss
        # B, _, H_ori, W_ori = data[0]['img'].size()
        # _, _, H_aug, W_aug = data[1]['img'].size()

        # ratio_ori_list, ratio_aug_list = [], []
        # for ix in range(len(data[0]['img_metas'])):
        #     ratio_ori_list.append((data[0]['img_metas'][ix]['img_shape'][1] / W_ori, data[0]['img_metas'][ix]['img_shape'][0] / H_ori))
        #     ratio_aug_list.append((data[1]['img_metas'][ix]['img_shape'][1] / W_aug, data[1]['img_metas'][ix]['img_shape'][0] / H_aug))

        # if self.distill_param_backbone > 0:
        #     consistency_backbone_loss = 0.
        #     for backbone_ori_ix, backbone_aug_ix in zip(backbone_ori, backbone_aug):
        #         loss_batch = 0.
        #         for batch_index in range(backbone_aug_ix.size(0)):
        #             b_ori_ix, b_aug_ix = backbone_ori_ix[[batch_index]], backbone_aug_ix[[batch_index]] 
                    
        #             # Original Image Extraction
        #             _, _, h_ori, w_ori = b_ori_ix.size()
        #             w_ori = int(w_ori * ratio_ori_list[batch_index][0])
        #             h_ori = int(h_ori * ratio_ori_list[batch_index][1])
        #             b_ori_ix = b_ori_ix[:, :, :h_ori, :w_ori]
                    
        #             # Augmentation Image Extraction
        #             _, _, h_aug, w_aug = b_aug_ix.size()
        #             w_aug = int(w_aug * ratio_aug_list[batch_index][0])
        #             h_aug = int(h_aug * ratio_aug_list[batch_index][1])
        #             b_aug_ix = F.interpolate(b_aug_ix[:, :, :h_aug, :w_aug], size=(h_ori, w_ori), mode='bilinear')
                    
        #             loss_batch += self.calc_consistency_loss(torch.unsqueeze(b_ori_ix.flatten(), dim=0), torch.unsqueeze(b_aug_ix.flatten(), dim=0))
                
        #         loss_batch /= B
        #         consistency_backbone_loss += loss_batch
            
        #     consistency_backbone_loss = consistency_backbone_loss * self.distill_param_backbone / len(backbone_ori)
        #     losses.update({'consistency_backbone_loss': consistency_backbone_loss})

    
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
        gt_feats_crop = gt_feats_crop.view(B, -1).clone().detach()
        
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
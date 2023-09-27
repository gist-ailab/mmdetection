from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
#from mmdet.registry import MODELS
import torch
import torch.nn.functional as F
from mmdet.models import build_detector
import copy
import numpy as np
from mmdet.core import bbox2roi
from .kd_trans import build_kd_trans, hcl

#@MODELS.register_module()
@DETECTORS.register_module()
class FasterRCNN_reviewKD_SCALE(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 teacher_cfg,
                 distill_name='',
                 distill_param=0.,
                 distill_param_backbone=0.,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_reviewKD_SCALE, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        teacher_cfg.model.type = 'FasterRCNNCont'
        teacher_cfg.model.roi_head.type = 'ContRoIHead'
        self.teacher_cfg = teacher_cfg

        self.distill_name = distill_name
        self.distill_param_backbone = distill_param_backbone
        self.distill_param = distill_param

        self.kd_trans = build_kd_trans(train_cfg)
        self.hcl = hcl

    def update_teacher(self, state_dict): 
        # Load Teacher Model
        self.teacher = build_detector(self.teacher_cfg.model,
                                      train_cfg=None,
                                      test_cfg=None)
        
        # Load Pretrained Teacher Weights
        self.teacher.load_state_dict(state_dict, strict=True)
        
        # Freeze Param
        for param in self.teacher.parameters():
            param.requires_grad = False

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        #feas_stu = [list(x)[0], list(x)[1], list(x)[2], list(x)[3]]

        if self.with_neck:
            x = self.neck(x)
        
        return x#, feas_stu


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
        self.teacher.eval()

        losses, gt_feats_down, backbone_down = self(**data[0])

        with torch.no_grad():
            _, _, backbone_teacher_down = self.teacher(**data[0])
            _, gt_feats_crop, backbone_crop = self.teacher(**data[1])
        ##############ReviewKD###############
        if 'ReviewKD' in self.distill_name:
            backbone_down = self.kd_trans(backbone_down)
            loss_mkdr = self.hcl(backbone_down, backbone_teacher_down) * 0.5
            losses.update({'reviewkd_loss': loss_mkdr})
        ####################################
        if 'F_SKD' in self.distill_name:
            # Backbone Feature Consistency Loss
            B, _, H_down, W_down = data[0]['img'].size()
            _, _, H_crop, W_crop = data[1]['img'].size()

            ratio_crop_list, ratio_down_list = [], []
            for ix in range(len(data[0]['img_metas'])):
                ratio_down_list.append((data[0]['img_metas'][ix]['img_shape'][1] / W_down, data[0]['img_metas'][ix]['img_shape'][0] / H_down))
                ratio_crop_list.append((data[1]['img_metas'][ix]['img_shape'][1] / W_crop, data[1]['img_metas'][ix]['img_shape'][0] / H_crop))

            if self.distill_param_backbone > 0:
                consistency_backbone_loss = 0.
                for backbone_down_ix, backbone_crop_ix in zip(backbone_down, backbone_crop):
                    loss_batch = 0.
                    for batch_index in range(backbone_down_ix.size(0)):
                        b_down_ix, b_crop_ix = backbone_down_ix[[batch_index]], backbone_crop_ix[[batch_index]] 
                        
                        # Cropped Image Extraction
                        _, _, h_crop, w_crop = b_crop_ix.size()
                        w_crop = int(w_crop * ratio_crop_list[batch_index][0])
                        h_crop = int(h_crop * ratio_crop_list[batch_index][1])
                        b_crop_ix = b_crop_ix[:, :, :h_crop, :w_crop]
                        
                        # Augmentation Image Extraction
                        _, _, h_down, w_down = b_down_ix.size()
                        w_down = int(w_down * ratio_down_list[batch_index][0])
                        h_down = int(h_down * ratio_down_list[batch_index][1])
                        b_down_ix = b_down_ix[:, :, :h_down, :w_down]
                        
                        # select cropped regions
                        _,_, h_imp, w_imp = b_down_ix.size()
                        x_l, x_u, y_l, y_u = data[1]['crop_loc'][batch_index]
                        b_down_ix = b_down_ix[:, :, int(h_imp * y_l) : int(h_imp * y_u), int(w_imp * x_l) : int(w_imp * x_u)]
                        
                        # interpolate to match size
                        b_down_ix = F.interpolate(b_down_ix, size=(h_crop, w_crop), mode='bilinear')
                        loss_batch += self.calc_consistency_loss(torch.unsqueeze(b_crop_ix.flatten(), dim=0), torch.unsqueeze(b_down_ix.flatten(), dim=0))
                    
                    loss_batch /= B
                    consistency_backbone_loss += loss_batch
                
                consistency_backbone_loss = consistency_backbone_loss * self.distill_param_backbone / len(backbone_crop)
                losses.update({'consistency_backbone_loss': consistency_backbone_loss})
        
        
            ## Calc Consistency Loss
            B = gt_feats_crop.size(0)
            gt_feats_crop = gt_feats_crop.view(B, -1)
            
            # select valid masks after crop
            crop_index = data[1]['crop_valid_inds']
            valid_index_list = []
            for ix in range(crop_index.size(0)):
                num = int(crop_index[ix][-1])
                valid_index_list.append(crop_index[ix][:num])
            valid_index_list = torch.cat(valid_index_list).bool()
            gt_feats_down = gt_feats_down[valid_index_list]
            gt_feats_down = gt_feats_down.view(B, -1)
            
            if self.distill_param > 0:
                consistency_rpn_loss = 0.
                positive_loss = self.calc_consistency_loss(gt_feats_crop, gt_feats_down)
                consistency_rpn_loss += positive_loss * self.distill_param
                losses.update({'consistency_rpn_loss': consistency_rpn_loss})



        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))

        return outputs
    
    def calc_consistency_loss(self, feat_teacher, feat_student):
        return torch.mean(1.0 - F.cosine_similarity(feat_student, feat_teacher))

    def min_max(self, x):
        max_value = torch.max(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        min_value = torch.min(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x_norm = (x - min_value) / (max_value - min_value)
        return x_norm

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x  = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    


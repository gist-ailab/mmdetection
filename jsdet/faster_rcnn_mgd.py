from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
#from mmdet.registry import MODELS
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
import copy
import numpy as np
from mmdet.core import bbox2roi
from mmdet.distillation.builder import DISTILLER,build_distill_loss
#from .mgdconnector import MGDConnector

        
def forward_train(self, feature):
    
    # if self.align is not None:
    #     feature = self.align(feature)

    N, C, H, W = feature.shape
    
    device = feature.device
    if not self.mask_on_channel:
        mat = torch.rand((N, 1, H, W)).to(device)
    else:
        mat = torch.rand((N, C, 1, 1)).to(device)

    mat = torch.where(mat > 1 - self.lambda_mgd,
                        torch.zeros(1).to(device),
                        torch.ones(1).to(device)).to(device)
    
    masked_fea = torch.mul(feature, mat)
    new_fea = self.generation(masked_fea)
    return new_fea

@DETECTORS.register_module()
class FasterRCNN_MGD(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 teacher_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN_MGD, self).__init__(
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
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.generation = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1))
        self.lambda_mgd = 0.45
        self.mask_on_channel = False

    def mask_generation(self, feature):
        
        # if self.align is not None:
        #     feature = self.align(feature)

        N, C, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                            torch.zeros(1).to(device),
                            torch.ones(1).to(device)).to(device)
        
        masked_fea = torch.mul(feature, mat)
        new_fea = self.generation(masked_fea)
    
        return new_fea
    
    def mask_generation_tea(self, feature):
        
        # if self.align is not None:
        #     feature = self.align(feature)

        N, C, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                            torch.zeros(1).to(device),
                            torch.ones(1).to(device)).to(device)
        
        masked_fea = torch.mul(feature, mat)
    
        return masked_fea
    
    def get_dis_loss(self, preds_S, preds_T):
        """Get MSE distance of preds_S and preds_T.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N, C, H, W = preds_T.shape
        dis_loss = self.loss_mse(preds_S, preds_T) / N

        return dis_loss
    
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
            #_, gt_feats_crop, backbone_crop = self.teacher(**data[1])
        
        #########################MGD##################################
        for i in range(0, len(backbone_down)):
            backbone_down_ = self.mask_generation(backbone_down[i])
            backbone_teacher_down_ = self.mask_generation_tea(backbone_teacher_down[i])
            dis_loss = self.get_dis_loss(backbone_down_, backbone_teacher_down_) * 0.0000005
            losses.update({'dis_loss_fpn{}'.format(i): dis_loss})
            #print('dis_loss_fpn{}'.format(i), dis_loss)
        ##############################################################
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))

        return outputs

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
    
    def calc_consistency_loss(self, feat_teacher, feat_student):
        return torch.mean(1.0 - F.cosine_similarity(feat_student, feat_teacher))

    def min_max(self, x):
        max_value = torch.max(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        min_value = torch.min(x.flatten(1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x_norm = (x - min_value) / (max_value - min_value)
        return x_norm

import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict




@DISTILLER.register_module()
class DetectionDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 fskd=False,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False):

        super(DetectionDistiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        self.use_fskd = fskd
        
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, data0, data1):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
            
        # data[0] : down-sampled / data[1] : cropped
        student_loss, gt_feats_down, backbone_down = self.student(**data0)
        with torch.no_grad():
            self.teacher.eval()
            _ = self.teacher(**data0)

        # FGD loss (for downsampled images)
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat, data0['gt_bboxes'], data0['img_metas'])
        
        # F-SKD Loss
        if self.use_fskd:
            # For High Resolution Teachers
            with torch.no_grad():
                _, gt_feats_crop, backbone_crop = self.teacher(**data1)
            
            # params
            self.distill_param_backbone = 1.0
            self.distill_param = 1.0
            
            # Backbone Feature Consistency Loss
            B, _, H_down, W_down = data0['img'].size()
            _, _, H_crop, W_crop = data1['img'].size()

            ratio_crop_list, ratio_down_list = [], []
            for ix in range(len(data0['img_metas'])):
                ratio_down_list.append((data0['img_metas'][ix]['img_shape'][1] / W_down, data0['img_metas'][ix]['img_shape'][0] / H_down))
                ratio_crop_list.append((data1['img_metas'][ix]['img_shape'][1] / W_crop, data1['img_metas'][ix]['img_shape'][0] / H_crop))

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
                    x_l, x_u, y_l, y_u = data1['crop_loc'][batch_index]
                    b_down_ix = b_down_ix[:, :, int(h_imp * y_l) : int(h_imp * y_u), int(w_imp * x_l) : int(w_imp * x_u)]
                    
                    # interpolate to match size
                    b_down_ix = F.interpolate(b_down_ix, size=(h_crop, w_crop), mode='bilinear')
                    loss_batch += self.calc_consistency_loss(torch.unsqueeze(b_crop_ix.flatten(), dim=0), torch.unsqueeze(b_down_ix.flatten(), dim=0))
                
                loss_batch /= B
                consistency_backbone_loss += loss_batch
            
            consistency_backbone_loss = consistency_backbone_loss * self.distill_param_backbone / len(backbone_crop)
            student_loss.update({'consistency_backbone_loss': consistency_backbone_loss})
        
        
            ## Calc Consistency Loss
            B = gt_feats_crop.size(0)
            gt_feats_crop = gt_feats_crop.view(B, -1)
            
            # select valid masks after crop
            crop_index = data1['crop_valid_inds']
            valid_index_list = []
            for ix in range(crop_index.size(0)):
                num = int(crop_index[ix][-1])
                valid_index_list.append(crop_index[ix][:num])
            valid_index_list = torch.cat(valid_index_list).bool()
            gt_feats_down = gt_feats_down[valid_index_list]
            gt_feats_down = gt_feats_down.view(B, -1)
            
            consistency_rpn_loss = 0.
            positive_loss = self.calc_consistency_loss(gt_feats_crop, gt_feats_down)
            consistency_rpn_loss += positive_loss * self.distill_param
            student_loss.update({'consistency_rpn_loss': consistency_rpn_loss})
        
        return student_loss
    
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
        losses = self(data[0], data[1])
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data[0]['img_metas']))

        return outputs

    def calc_consistency_loss(self, feat_teacher, feat_student):
        return torch.mean(1.0 - F.cosine_similarity(feat_student, feat_teacher))
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


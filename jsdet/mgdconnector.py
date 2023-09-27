# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

class MGDConnector(nn.Module):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(
        self,
        student_channels = 256,
        teacher_channels = 256,
        lambda_mgd = 0.65,
        mask_on_channel= False,
        init_cfg = None):
        super(MGDConnector, self).__init__()
        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels 
        self.generation = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1))
        
    def forward(self, feature):
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        return self.forward_train(feature)
    
    def forward_train(self, feature):
        
        # if self.align is not None:
        #     feature = self.align(feature)

        N, C, H, W = feature.shape
        print(N, C, H, W)
        exit()
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
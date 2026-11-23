from mmengine import MODELS
import torch
from mmengine.model import BaseModule
from .depthnet import DepthNet
from .unet2d import UNet2D
import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

@MODELS.register_module()
class GaussianDepthBranch(BaseModule):
    def __init__(
        self,
    ):
        super().__init__()
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.depthanything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 10})
        self.depthanything.load_state_dict(torch.load('/home/gl/zph_ws/EmbodiedOcc/checkpoints/finetune_scannet_depthanythingv2.pth'))
        
        self.unet2d = UNet2D.build(out_feature=200, use_decoder=True, frozen_encoder=True)
        self.depthnet = DepthNet(201, 256, 200, 64)


    def forward(self, imgs, metas):
        return None

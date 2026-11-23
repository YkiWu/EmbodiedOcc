import torch
import numpy as np
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/Depth-Anything-V2/metric_depth')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/model/depthbranch')
from depth_anything_v2.dpt import DepthAnythingV2
from depthnet import DepthNet
from unet2d import DecoderBN
import torch.nn as nn
from PIL import Image
import cv2
import torch.nn.functional as F

@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        flag_depthbranch=False,
        flag_depthanything_as_gt=False,
        depthbranch=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.flag_depthbranch = flag_depthbranch
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                # depth branch
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
                checkpoint = torch.load('/home/gl/zph_ws/EmbodiedOcc/checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_key = k[len('module.'):] 
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                self.depthanything.load_state_dict(new_state_dict)
            
            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            # basemodel = torch.hub.load(
            #     "/home/gl/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            # )
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
            
            self.backbone = basemodel
            
            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )
        else:
            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            basemodel = torch.hub.load(
                "/home/gl/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
            
            self.backbone = basemodel
            
            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if future_decoder is not None: 
            self.future_decoder = MODELS.build(future_decoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/gl/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640
        
        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)
            
        img_feats_backbone = feature_x
        
        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone) # dict
        
        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
                
        return img_feats_reshaped, img_feats_out['1_1'] # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
    
    def obtain_bev(self, imgs, metas):
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        
        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs) # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
        
        if self.flag_depthbranch:
            if self.flag_depthanything_as_gt:
                # depth branch
                self.depthanything.eval()
                image_ = metas[0]['img_depthbranch']
                depth_pred = self.depthanything.infer_image(image_, 480, 640, 480)
                depthnet_output = depth_pred
            else:  
                depthnet_output = None
        else:
            depthnet_output = None
        
        anchor, instance_feature, depth2occ, depthnet_output_loss, predtoreturn = self.lifter(self.flag_depthbranch, self.flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas)    # b, g, c 
        anchor = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # b, g, c
        
        return anchor, depth2occ, depthnet_output_loss, predtoreturn
    
    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < F
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
        else:
            bev, depth2occ, depthnet_output_loss, predtoreturn = self.obtain_bev(imgs, metas)

        # BF, H, W, C = bev.shape
        BF, G, C = bev.shape # bev is actually anchors [1, 21600, 24]
        bev = bev.reshape(B, F, G, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()
        output_dict = self.head(
            bev_feat=bev_predict,  # [1, 1, 21600, 24]
            points=points, 
            label=label, 
            output_dict=output_dict, 
            metas=metas,
            test_mode=test_mode)
        
        return output_dict, depth2occ, predtoreturn
        
    def frame_split(self, grad_frames, imgs, metas):
        F = imgs.shape[1]
        index = np.random.permutation(F)
        inv_index = np.argsort(index)
        imgs_grad = imgs[:, index[:grad_frames]]
        imgs_no_grad = imgs[:, index[grad_frames:]]
        metas_grad = deepcopy(metas)
        metas_no_grad = deepcopy(metas)
        for meta, meta_grad, meta_no_grad in zip(metas, metas_grad, metas_no_grad):
            lidar2img = np.asarray(meta['lidar2img'])
            meta_grad['lidar2img'] = lidar2img[index[:grad_frames]]
            meta_no_grad['lidar2img'] = lidar2img[index[grad_frames:]]
            img_aug_matrix = meta['img_aug_matrix']
            meta_grad['img_aug_matrix'] = img_aug_matrix[index[:grad_frames]]
            meta_no_grad['img_aug_matrix'] = img_aug_matrix[index[grad_frames:]]

        return imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index
    
    def forward_autoreg(self,
                        imgs=None,
                        metas=None,
                        points=None,
                        label=None,
                        test_mode=True,
                        **kwargs,
        ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        
        bev = self.obtain_bev(imgs, metas)
        BF, G, C = bev.shape # bev is actually anchors
        bev = bev.reshape(B, F, G, C)

        output_dict = self.future_decoder.forward_autoreg(bev, metas)
        bev_predict = output_dict.pop('bev')
        output_dict = self.head(
            bev_feat=bev_predict, 
            points=points, 
            label=label, 
            output_dict=output_dict, 
            metas=metas,
            test_mode=test_mode)

        return output_dict

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix
import sys

@MODELS.register_module()
class GaussianOccHeadGlobal(BaseModule):
    def __init__(
        self,
        empty_label=17, # 12
        num_classes=18, # 13
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        with_empty=False,
        empty_args=dict(),
        pc_range=[],
        scale_range=[],
        include_opa=True,
        semantics_activation='softmax'
    ):
        super().__init__()

        self.empty_label = empty_label
        self.num_classes = num_classes
        self.classes = list(range(num_classes))

        self.cuda_kwargs = cuda_kwargs
        sys.path.append('/home/gl/zph_ws/EmbodiedOcc/model/head/gaussian_occ_head/ops/localagg')
        from local_aggregate import LocalAggregator
        self.aggregator = LocalAggregator(**cuda_kwargs)
        
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.include_opa = include_opa
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = self.num_classes if not with_empty else self.num_classes - 1
        self.semantics_activation = semantics_activation
    
    def anchor2gaussian(self, anchor, metas):
        
        xyz = anchor[..., :3]
        
        gs_scales = anchor[..., 3:6]
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        
        rot = anchor[..., 6: 10]
        
        opas = anchor[..., 10: (10 + int(self.include_opa))]
        shs = torch.zeros(*anchor.shape[:-1], 0, device=anchor.device, dtype=anchor.dtype)
        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
            
        # semantics = torch.cat([semantics, torch.zeros_like(semantics[..., :1])], dim=-1)
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics
        )
        return gaussian
    
    def prepare_gaussian_args(self, gaussians, metas, v_origin, s_size):
        
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            vox_origin = metas['global_scene_origin']
            scene_size = metas['global_scene_size']
            # vox_origin = v_origin
            # scene_size = s_size
            vox_center = vox_origin + scene_size / 2
            self.empty_mean = vox_center[None, None, :].to(torch.float32)
            # self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            
            # opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1) # FIXME
            opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1) # FIXME
            
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)

        bs, g, _ = means.shape

        S = torch.zeros(bs, g, 3, 3, dtype=rotations.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]

        R = get_rotation_matrix(rotations) # b, g, 3, 3

        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        
        CovInv = Cov.float().inverse() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv
    
    def prepare_gt_xyz(self, metas, tensor):
        
        gt_xyz = metas['global_pts'].unsqueeze(0)
        
        return gt_xyz


    def forward_gaussian(self, bev_feat, points, label, output_dict, metas, test_mode=False, label_mask=None, v_origin=None, s_size=None):
        
        assert self.cuda_kwargs['H'] >= metas['global_scene_dim'][0]
        assert self.cuda_kwargs['W'] >= metas['global_scene_dim'][1]
        assert self.cuda_kwargs['D'] >= metas['global_scene_dim'][2]
       
        assert bev_feat.shape[0] == 1
        anchors = bev_feat # [1, 1, N, 23]
        gt_xyz = self.prepare_gt_xyz(metas, anchors).flatten(0, 1).unsqueeze(0) # bf, x, y, z, 3 

        B, F, G, _ = anchors.shape
        anchors = anchors.flatten(0, 1) # [1, N, 23]
        gaussians = self.anchor2gaussian(anchors, metas)
        
        return gaussians
    
    def forward(self, bev_feat, points, label, output_dict, metas, test_mode=False, label_mask=None, v_origin=None, s_size=None):
        
        assert self.cuda_kwargs['H'] >= metas['global_scene_dim'][0]
        assert self.cuda_kwargs['W'] >= metas['global_scene_dim'][1]
        assert self.cuda_kwargs['D'] >= metas['global_scene_dim'][2]
       
        assert bev_feat.shape[0] == 1
        anchors = bev_feat # [1, 1, N, 23]
        gt_xyz = self.prepare_gt_xyz(metas, anchors).flatten(0, 1).unsqueeze(0) # bf, x, y, z, 3 

        B, F, G, _ = anchors.shape
        anchors = anchors.flatten(0, 1) # [1, N, 23]
        gaussians = self.anchor2gaussian(anchors, metas)
        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians, metas, v_origin, s_size)
        
        sampled_xyz = gt_xyz.flatten(1, 3).float()
        origi_opa = origi_opa.flatten(1, 2)
        
        semantics = []
        nyu_pc_min = metas['global_scene_origin']
        nyu_pc_max = nyu_pc_min + metas['global_scene_size']
        
        epsilon = 1e-3
        mask = (means[..., 0] > (nyu_pc_min[0]+epsilon)) & (means[..., 0] < (nyu_pc_max[0]-epsilon)) & (means[..., 1] > (nyu_pc_min[1]+epsilon)) & (means[..., 1] < (nyu_pc_max[1]-epsilon)) & (means[..., 2] > (nyu_pc_min[2]+epsilon)) & (means[..., 2] < (nyu_pc_max[2]-epsilon))
        means = means[mask].unsqueeze(0)
        origi_opa = origi_opa[mask].unsqueeze(0)
        opacities = opacities[mask].unsqueeze(0)
        scales = scales[mask].unsqueeze(0)
        CovInv = CovInv[mask].unsqueeze(0)
        
        origin_use = metas['global_scene_origin'].to(torch.float32).to(means.device)
        for i in range(len(sampled_xyz)):
            semantic = self.aggregator(
                sampled_xyz[i:(i+1)], 
                means[i:(i+1)], 
                origi_opa[i:(i+1)],
                opacities[i:(i+1)],
                scales[i:(i+1)],
                CovInv[i:(i+1)],
                metas,
                origin_use) # n, c
            semantics.append(semantic)

        semantics = torch.stack(semantics, dim=0).transpose(1, 2) # [1, 13, 129600]
        spatial_shape = label.shape[2:] # [60, 60, 36]
        
        semantics = semantics.unflatten(-1, spatial_shape)
        semantics = semantics.argmax(dim=1).long()
        semantics = semantics.squeeze(0)
        
        label[label == 0] = 12
        label = label.squeeze(0).squeeze(0)
        
        result_dict = {
            'predict': semantics, # x_dim, y_dim, z_dim
            'label': label,       
            'mask': label_mask        
        }
        
        output_dict.update(result_dict)
        
        output_dict.update({
                'gaussians': gaussians
            })
        return output_dict
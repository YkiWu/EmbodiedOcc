import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix, safe_get_quaternion, batch_quaternion_multiply
import sys

@MODELS.register_module()
class GaussianOccHeadLocal(BaseModule):
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
        
        
        # myfix
        cam_vox_range = metas[0]['cam_vox_range'].to(anchor.device)
        xyz = cartesian(anchor, cam_vox_range)
        # endfix
        
        # xyz = cartesian(anchor, nyu_pc_range)
        gs_scales = safe_sigmoid(anchor[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = anchor[..., 6: 10]
        opas = safe_sigmoid(anchor[..., 10: (10 + int(self.include_opa))])
        shs = torch.zeros(*anchor.shape[:-1], 0, device=anchor.device, dtype=anchor.dtype)
        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics
        )
        return gaussian
    
    def gaussian2vis(self, gaussians, metas):
        means = gaussians.means
        b_, g_, _ = means.shape
        means = means.reshape(-1, 3)
        means_cam = torch.cat((means, torch.ones((means.shape[0], 1), device=means.device)), dim=1).to(torch.float32)
        cam2world = metas[0]['cam2world'].to(torch.float32)
        means_world_ = (cam2world @ means_cam.unsqueeze(-1)).squeeze(-1)
        means_world = means_world_[:, :3]
        means_world = means_world.reshape(b_, g_, 3)
        means = means_world
        nyu_pc_min = metas[0]['vox_origin']
        nyu_pc_max = nyu_pc_min + metas[0]['scene_size']
        epsilon = 1e-3
        mask_toreturn = (means_world[..., 0] > (nyu_pc_min[0]+epsilon)) & (means_world[..., 0] < (nyu_pc_max[0]-epsilon)) & (means_world[..., 1] > (nyu_pc_min[1]+epsilon)) & (means_world[..., 1] < (nyu_pc_max[1]-epsilon)) & (means_world[..., 2] > (nyu_pc_min[2]+epsilon)) & (means_world[..., 2] < (nyu_pc_max[2]-epsilon))
        means_to_return = means_world[mask_toreturn].unsqueeze(0)
        
        scales = gaussians.scales
        scales_to_return = scales[mask_toreturn].unsqueeze(0)
        
        rotations = gaussians.rotations
        rotations_cam = rotations.squeeze(0) # N, 4
        c2w_rot = metas[0]['cam2world'][:3, :3].to(torch.float32)
        c2w_quat = safe_get_quaternion(c2w_rot.unsqueeze(0)).squeeze(0)
        rotations_world = batch_quaternion_multiply(c2w_quat, rotations_cam).unsqueeze(0) # 1, N, 4
        rotations_to_return = rotations_world[mask_toreturn].unsqueeze(0)
        
        harmonics = gaussians.harmonics
        harmonics_to_return = harmonics[mask_toreturn].unsqueeze(0)
        
        opacities = gaussians.opacities
        opacities_to_return = opacities[mask_toreturn].unsqueeze(0)
        
        semantics = gaussians.semantics
        semantics = torch.cat([semantics, torch.zeros_like(semantics[..., :1])], dim=-1)
        semantics_to_return = semantics[mask_toreturn].unsqueeze(0)
        
        gaussian = GaussianPrediction(
            means=means_to_return,
            scales=scales_to_return,
            rotations=rotations_to_return,
            harmonics=harmonics_to_return,
            opacities=opacities_to_return,
            semantics=semantics_to_return
        )
        
        return gaussian
        
    
    def prepare_gaussian_args(self, gaussians, metas, anchor_new_tag, instance_feature_cache):
        
        means = gaussians.means # b, g, 3
        # myfix
        b_, g_, _ = means.shape
        means = means.reshape(-1, 3)
        means_cam = torch.cat((means, torch.ones((means.shape[0], 1), device=means.device)), dim=1).to(torch.float32)
        cam2world = metas[0]['cam2world'].to(torch.float32)
        means_world_ = (cam2world @ means_cam.unsqueeze(-1)).squeeze(-1)
        means_world = means_world_[:, :3]
        means_world = means_world.reshape(b_, g_, 3)
        means = means_world
        
        nyu_pc_min = metas[0]['vox_origin']
        nyu_pc_max = nyu_pc_min + metas[0]['scene_size']
        epsilon = 1e-3
        mask_toreturn = (means_world[..., 0] > (nyu_pc_min[0]+epsilon)) & (means_world[..., 0] < (nyu_pc_max[0]-epsilon)) & (means_world[..., 1] > (nyu_pc_min[1]+epsilon)) & (means_world[..., 1] < (nyu_pc_max[1]-epsilon)) & (means_world[..., 2] > (nyu_pc_min[2]+epsilon)) & (means_world[..., 2] < (nyu_pc_max[2]-epsilon))
        means_to_return = means_world[mask_toreturn] # mask_toreturn [1, N]  -> [N', 3]
        anchor_new_tag = anchor_new_tag[mask_toreturn] # N'
        instance_feature_cache = instance_feature_cache[mask_toreturn] 
        
        # endfix
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        
        scales_to_return = (scales - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0]) # b, g, 3
        scales_to_return = scales_to_return[mask_toreturn] # N', 3
        opacities_to_return = opacities[mask_toreturn] # N', 12
        origi_opa_to_return = origi_opa[mask_toreturn] # N', 1
        rotations_cam = rotations.squeeze(0) # N, 4
        c2w_rot = metas[0]['cam2world'][:3, :3].to(torch.float32)
        c2w_quat = safe_get_quaternion(c2w_rot.unsqueeze(0)).squeeze(0)
        rotations_world = batch_quaternion_multiply(c2w_quat, rotations_cam).unsqueeze(0) # 1, N, 4
        rotations_to_return = rotations_world[mask_toreturn] # N', 4
        
        # fov
        world2cam = metas[0]['world2cam'].to(torch.float32)
        cam_k = metas[0]['cam_k'].to(torch.float32)
        means_to_return_true = means_to_return.to(torch.float32)
        means_to_return_world = torch.cat((means_to_return, torch.ones((means_to_return.shape[0], 1), device=means_to_return.device)), dim=1).to(torch.float32)
        means_to_return_cam = (world2cam @ means_to_return_world.unsqueeze(-1)).squeeze(-1)
        means_cam_x = means_to_return_cam[:, 0]
        means_cam_y = means_to_return_cam[:, 1]
        means_cam_z = means_to_return_cam[:, 2]
        
        means_cam_mask = means_cam_z > 1e-6
        means_cam_x = means_cam_x[means_cam_mask]
        means_cam_y = means_cam_y[means_cam_mask]
        means_cam_z = means_cam_z[means_cam_mask]
        means_to_return_true = means_to_return_true[means_cam_mask]
        scales_to_return = scales_to_return[means_cam_mask]
        rotations_to_return = rotations_to_return[means_cam_mask]
        origi_opa_to_return = origi_opa_to_return[means_cam_mask]
        opacities_to_return = opacities_to_return[means_cam_mask]
        anchor_new_tag = anchor_new_tag[means_cam_mask]
        instance_feature_cache = instance_feature_cache[means_cam_mask]
        
        means_cam_z = torch.clamp(means_cam_z, min=1e-6)
        means_cam_x = means_cam_x / means_cam_z
        means_cam_y = means_cam_y / means_cam_z
        
        means_pix_x = torch.floor(means_cam_x * cam_k[0, 0] + cam_k[0, 2])
        means_pix_y = torch.floor(means_cam_y * cam_k[1, 1] + cam_k[1, 2])
        means_pix_mask = (means_pix_x >= 0) & (means_pix_x < 640) & (means_pix_y >= 0) & (means_pix_y < 480)
        means_to_return_true = means_to_return_true[means_pix_mask]
        scales_to_return = scales_to_return[means_pix_mask]
        rotations_to_return = rotations_to_return[means_pix_mask]
        origi_opa_to_return = origi_opa_to_return[means_pix_mask]
        opacities_to_return = opacities_to_return[means_pix_mask]
        anchor_new_tag = anchor_new_tag[means_pix_mask]
        instance_feature_cache = instance_feature_cache[means_pix_mask]
        
        gaussianstensor_to_return = torch.cat([means_to_return_true, scales_to_return, rotations_to_return, origi_opa_to_return, opacities_to_return], dim=-1).unsqueeze(0) # 1, N', 23
        gaussiantensor_to_return_tag = torch.ones_like(gaussianstensor_to_return[..., :1], dtype=torch.float32)
        gaussian_return_splat_tag = torch.ones_like(gaussianstensor_to_return[..., :1], dtype=torch.float32)
        gaussianstensor_to_return = torch.cat([gaussianstensor_to_return, gaussiantensor_to_return_tag], dim=-1)
        gaussianstensor_to_return = torch.cat([gaussianstensor_to_return, gaussian_return_splat_tag], dim=-1)
        instance_feature_cache = instance_feature_cache.unsqueeze(0)
        
        # myfix:use fov to splat
        means = means[mask_toreturn].unsqueeze(0)
        scales = scales[mask_toreturn].unsqueeze(0)
        rotations = rotations[mask_toreturn].unsqueeze(0)
        opacities = opacities[mask_toreturn].unsqueeze(0)
        origi_opa = origi_opa[mask_toreturn].unsqueeze(0)
        # endfix
        
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            vox_origin = metas[0]['vox_origin']
            scene_size = metas[0]['scene_size']
            vox_center = vox_origin + scene_size / 2
            self.empty_mean = vox_center[None, None, :]
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

        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]

        R = get_rotation_matrix(rotations) # b, g, 3, 3
        
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        
        # myfix
        c2w_rot = metas[0]['cam2world'][:3, :3]
        c2w_rot_T = metas[0]['cam2world'][:3, :3].T
        c2w_rot = c2w_rot.unsqueeze(0).unsqueeze(0).repeat(bs, g, 1, 1).to(torch.float32)
        c2w_rot_T = c2w_rot_T.unsqueeze(0).unsqueeze(0).repeat(bs, g, 1, 1).to(torch.float32)
        Cov = torch.matmul(c2w_rot, torch.matmul(Cov, c2w_rot_T))
        # endfix
        
        CovInv = Cov.float().cpu().inverse().cuda() # b, g, 3, 3
        
        return means, origi_opa, opacities, scales, CovInv, gaussianstensor_to_return, anchor_new_tag.unsqueeze(0), instance_feature_cache
    
    def prepare_gt_xyz(self, metas, tensor):
        
        gt_xyz = metas[0]['occ_xyz'].unsqueeze(0)
        
        return gt_xyz

    def forward(self, bev_feat, points, label, output_dict, metas, test_mode=False, anchor_new_tag=None, instance_feature_cache=None):
        # means3D: 
        # gt_xyz: b, x, y, z, 3
        # gt_label: b, x, y, z

        # sampled_xyz: b, n, 3
        # sampled_label: b, n
        assert bev_feat.shape[0] == 1
        anchors = bev_feat # [1, 1, 21600, 23]
        gt_xyz = self.prepare_gt_xyz(metas, anchors).flatten(0, 1).unsqueeze(0) # bf, x, y, z, 3 [1, 60, 60, 36, 3]

        B, F, G, _ = anchors.shape
        anchors = anchors.flatten(0, 1) # [1, 21600, 24]
        gaussians = self.anchor2gaussian(anchors, metas)
        means, origi_opa, opacities, scales, CovInv, gaussianstensor_to_return, anchor_new_tag, instance_feature_toreturn = self.prepare_gaussian_args(gaussians, metas, anchor_new_tag, instance_feature_cache)
        gaussians_to_vis = self.gaussian2vis(gaussians, metas)
        
        sampled_xyz = gt_xyz.flatten(1, 3).float()
        origi_opa = origi_opa.flatten(1, 2)
        
        semantics = []
        nyu_pc_min = metas[0]['vox_origin']
        nyu_pc_max = nyu_pc_min + metas[0]['scene_size']
        
        epsilon = 1e-3
        mask = (means[..., 0] > (nyu_pc_min[0]+epsilon)) & (means[..., 0] < (nyu_pc_max[0]-epsilon)) & (means[..., 1] > (nyu_pc_min[1]+epsilon)) & (means[..., 1] < (nyu_pc_max[1]-epsilon)) & (means[..., 2] > (nyu_pc_min[2]+epsilon)) & (means[..., 2] < (nyu_pc_max[2]-epsilon))
        
        means = means[mask].unsqueeze(0)
        origi_opa = origi_opa[mask].unsqueeze(0)
        opacities = opacities[mask].unsqueeze(0)
        scales = scales[mask].unsqueeze(0)
        CovInv = CovInv[mask].unsqueeze(0)
        
        origin_use = metas[0]['vox_origin'].to(torch.float32).to(means.device)
        
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
        
        result_dict = {
            'ce_input': semantics.unflatten(-1, spatial_shape), # [1, 13, 60, 60, 36]
            'ce_label': label.squeeze(0),                       # [1, 60, 60, 36]
            'fov_mask': metas[0]['fov_mask'],                   # [60, 60, 36]
        }
        # import pdb; pdb.set_trace()
        output_dict.update(result_dict)
        output_dict.update({
                'gaussians': gaussians
            })
        return output_dict, gaussianstensor_to_return, instance_feature_toreturn, gaussians_to_vis
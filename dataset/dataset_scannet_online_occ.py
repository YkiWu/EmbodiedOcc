import os
import json
import glob
import numpy as np
import numba as nb
import torch
from torch.utils import data
import pickle
from PIL import Image
from mmcv.image.io import imread
import copy
from pyquaternion import Quaternion
from . import OPENOCC_DATASET
from dataset.nyu_utils import vox2pix
from torchvision import transforms
from mmcv.image.io import imread
import math, cv2
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet

@OPENOCC_DATASET.register_module()
class Scannet_Online_SceneOcc_Dataset(data.Dataset):
    def __init__(
        self,
        num_frames=30, # 一个场景的有效图片数量
        grid_size_occ=[60, 60, 36],
        empty_idx=0,
        phase='train',
        num_pts=21600,
        data_tag='base',
        ):
        
        self.occscannet_root = '/home/gl/zph_ws/EmbodiedOcc/data/scene_occ'
        self.posed_images_root = '/home/gl/zph_ws/EmbodiedOcc/data/occscannet/posed_images'
        self.phase = phase
        self.num_frames = num_frames
        self.grid_size_occ = grid_size_occ # local size
        self.empty_idx = empty_idx
        self.data_tag = data_tag
        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        
        if data_tag == 'base':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_online.txt' # scenexxxx_xx
        elif data_tag == 'mini':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_mini_online.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = self.used_subscenes[i].strip() # scenexxxx_xx
        
        self.num_pts = num_pts
        
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.used_subscenes)

    def __getitem__(self, index):
        name = self.used_subscenes[index] # scenexxxx_xx
        meta = {}
        meta['scene_name'] = name
        
        # load global infos
        scene_pkg_pth = f'{self.occscannet_root}/global_occ_package/{name}.pkl'
        with open(scene_pkg_pth, 'rb') as f:
            scene_pkg = pickle.load(f)
        meta['global_scene_dim'] = scene_pkg['scene_dim']
        meta['global_scene_size'] = 0.08 * np.array(scene_pkg['scene_dim']) # (x_dim, y_dim, z_dim) * 0.08
        meta['global_labels'] = scene_pkg['global_labels'] # (x_dim, y_dim, z_dim)
        meta['global_pts'] = scene_pkg['global_pts'] # (x_dim, y_dim, z_dim, 3)
        meta['global_scene_origin'] = np.array([scene_pkg['global_pts'][:, :, :, 0].min(), scene_pkg['global_pts'][:, :, :, 1].min(), scene_pkg['global_pts'][:, :, :, 2].min()])
        valid_img_paths = scene_pkg['valid_img_paths']
        valid_img_paths = [os.path.join(self.posed_images_root, p.split('/')[-2], p.split('/')[-1]) for p in valid_img_paths]
        sorted_image_paths = sorted(
            valid_img_paths,
            key=lambda x: int(x.split("/")[-1].split(".")[0])
        )
        meta['valid_img_paths'] = sorted_image_paths
        meta['global_mask'] = scene_pkg['global_mask'].astype(np.bool_) # (x_dim, y_dim, z_dim)
        
        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # use for loop to load local infos
        monometa_list = []
        N_img = []
        N_occ = []
        for i in range(len(sorted_image_paths)):
            monometa = {}
            
            monometa['global_scene_origin'] = meta['global_scene_origin']
            monometa['global_scene_size'] = meta['global_scene_size']
            
            rgb_path = sorted_image_paths[i]
            depth_path = rgb_path.replace('jpg', 'png')
            img_idx = rgb_path.split("/")[-1].split(".")[0] # 'xxxxx'
            this_name = meta['scene_name'] + '/' + img_idx
            monometa['name'] = this_name # 'scene0000_00/00000'
            
            # 加载global2local的occ
            my_pth = self.occscannet_root + '/streme_occ_new_package/' + self.phase + '/' + meta['scene_name'] + '_' + img_idx + '_new.pkl'
            with open(my_pth, 'rb') as f1:
                data1 = pickle.load(f1)
                my_target = data1['local_label']
                mask_in_global_from_this = data1['mask_in_global']
            
            monometa['mask_in_global_from_this'] = mask_in_global_from_this
            
            mono_pkg_pth = f'/home/gl/zph_ws/EmbodiedOcc/data/occscannet/gathered_data/{this_name}.pkl'
            with open(mono_pkg_pth, 'rb') as f:
                data = pickle.load(f)
            monometa['scene_size'] = self.scene_size
            cam_pose = data['cam_pose']
            monometa['cam2world'] = cam_pose
            world2cam = np.linalg.inv(cam_pose)
            monometa['world2cam'] = world2cam
            
            img_depthbranch = cv2.imread(rgb_path)
            img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
            img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
            sample = transform({'image': img_depthbranch})
            img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
            monometa['img_depthbranch'] = img_depthbranch
            monometa['rgb_path'] = rgb_path
            monometa['depth_path'] = depth_path
            
            
            this_img = imread(rgb_path, 'unchanged').astype(np.float32)
            this_H, this_W, _ = this_img.shape
            new_H, new_W = 480, 640
            # resize
            new_img = cv2.resize(this_img, (new_W, new_H))
            W_factor = new_W / this_W
            H_factor = new_H / this_H
            N_img.append(new_img)
            # img = np.stack(N_img, 0) # [1, 968, 1296, 3]
            this_H, this_W = new_H, new_W
            # img = [img] # [1, 1, 968, 1296, 3]

            cam_intrin = data['intrinsic']
            cam_intrin[0, 0] *= W_factor
            cam_intrin[0, 2] *= W_factor
            cam_intrin[1, 1] *= H_factor
            cam_intrin[1, 2] *= H_factor
            
            monometa['cam_k'] = cam_intrin[:3, :3]
            viewpad = np.eye(4)
            viewpad[:monometa['cam_k'].shape[0], :monometa['cam_k'].shape[1]] = monometa['cam_k']
            monometa['cam2img'] = viewpad
            world2img = (viewpad @ world2cam)
            monometa['world2img'] = world2img
            depth_gt = Image.open(depth_path).convert('I;16')
            depth_gt = np.array(depth_gt) / 1000.0
            monometa['depth_gt'] = depth_gt
            
            vox_origin = data["voxel_origin"]
            monometa['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
            # target = data["target_1_4"] # 60, 60, 36
            target = my_target
            target = np.transpose(target, (1, 0, 2))
            # 把代表unknown的255换成0，把代表空的0换成12
            target[target == 0] = 12
            target[target == 255] = 0 
            occ = target # (60, 60, 36)
            nonemptymask = (occ != 12)
            N_occ.append(occ)
            # occ = [occ] # [1, 60, 60, 36]
            
            projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
                world2cam,
                monometa['cam_k'],
                monometa['vox_origin'],
                self.voxel_size,
                this_W,
                this_H,
                self.scene_size,
                dim_60_60_36=True,
            )
            
            monometa['projected_pix'] = projected_pix
            monometa['fov_mask'] = fov_mask.reshape(60, 60, 36)
            
            monometa['pix_z'] = pix_z
            monometa['occ_xyz'] = occ_xyz.reshape(60, 60, 36, 3)
            
            vox_near = monometa['vox_origin']
            vox_far = vox_near + monometa['scene_size']
            nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
            monometa['nyu_pc_range'] = nyu_pc_range
            monometa['num_depth'] = self.num_pts
            
            cam_vox_near = np.array([-5, -6, -3])
            cam_vox_far = np.array([5, 6, 8])
            cam_vox_range = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)
            monometa['cam_vox_range'] = cam_vox_range
            
            monometa['occ_mask_valid'] = (occ != 0)
            # import pdb; pdb.set_trace()
            # monometa['occ_mask_valid_fov'] = (occ != 0) & fov_mask
            monometa['label'] = occ
            
            monometa_list.append(monometa)
        
        meta['monometa_list'] = monometa_list

        img = np.stack(N_img, 0)
        img = [img]
        imgs = np.stack(img, 0)
        
        occs = np.stack(N_occ, 0)
        data_tuple = (imgs, meta, occs)
        return data_tuple

    def get_meshgrid(self, ranges, grid, reso):
        pass
    
    def get_data_info(self, info):
        pass

    def get_scene_index(self, scene_name=None):
        pass
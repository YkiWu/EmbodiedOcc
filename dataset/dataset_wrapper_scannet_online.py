import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D


img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Online_SceneOcc_DatasetWrapper(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=True),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=False),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        imgs, metas, occ = data
        
        # deal with img augmentation
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
        FN, C, H, W = imgs.shape
        imgs = imgs.reshape(F, N, C, H, W)
        
        for i in range(N):
            metas['monometa_list'][i]['img_shape'] = [imgs_dict['img_shape'][i]]
        
        if imgs_dict.get('img_aug_matrix'):
            for i in range(N):
                metas['monometa_list'][i]['img_aug_matrix'] = imgs_dict['img_aug_matrix'][i][None, None]
        
        data_tuple = (imgs, metas, occ)
        return data_tuple
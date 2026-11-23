import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=(2560, 1440))
    display.start()
    offscreen = True

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from mayavi import mlab
import mayavi
import cv2
import open3d as o3d
from utils.iou_as_iso import SSCMetrics
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

# torchrun --nproc_per_node=1 vis_embodied.py
from utils.iou_eval import IOUEvalBatch
from utils.iou_as_iso import SSCMetrics
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_notddp

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
from mmengine.registry import MODELS
import open3d as o3d

from matplotlib import pyplot as plt, cm, colors
from pyquaternion import Quaternion

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/Depth-Anything-V2/metric_depth')
from PIL import Image

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw_gaussian(vox_near, scene_size, save_dir, gaussian, idx=0, sem=False, K_idx=0, scene_name=None):
    empty_label = 12
    sem_cmap = np.array(
        [
            [  0,   0,   0, 255],       # others
            [214,  38, 40, 255],       # "ceiling"             orange
            [43, 160, 4, 255],       # "floor"              pink
            [158, 216, 229, 255],       # "wall"                  yellow
            [  114, 158, 206, 255],       # "window"                  blue
            [  204, 204, 91, 255],       # "chair"  cyan
            [255, 186, 119, 255],       # "bed"           dark orange
            [147, 102, 188, 255],       # "sofa"           red
            [30, 119, 181, 255],       # "table"         light yellow
            [160, 188, 33, 255],       # "tvs"              brown
            [255, 127, 12, 255],       # "furn"                purple                
            [196, 175, 214, 255],       # "objs"   dark pink
        ]
    ).astype(np.float32) / 255.
    
    means = gaussian.means[idx].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[idx].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[idx].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[idx]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[idx][..., :1]) 
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[idx].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    nonempty_mask = pred != empty_label
    
    mask = nonempty_mask
    print('number of nonempty gaussians: ', mask.sum())

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=90, azim=90) # FIXME
    # ax.view_init(elev=elevation+45, azim=azimuth) # FIXME

    scalar = 0.8

    # compute each and plot each ellipsoid iteratively
    
    vox_far = vox_near + scene_size
    nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
    
    border = np.array([
        [vox_near[0], vox_near[1], 0.0],
        [vox_near[0], vox_far[1], 0.0],
        [vox_far[0], vox_near[1], 0.0],
        [vox_far[0], vox_far[1], 0.0],
    ])
    
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        if not sem:
            ax.plot_surface(
                xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
                rstride=1, cstride=1, color=m.to_rgba(center[2]), linewidth=0, alpha=opas[indx], shade=True)
        else:
            ax.plot_surface(
                # xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
                xyz[..., 0], xyz[..., 1], xyz[..., 2], 
                rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("auto")
    ax.grid(False)
    ax.set_axis_off()    

    filepath = os.path.join(save_dir, scene_name+f'_{K_idx}_gauss.png')
    plt.savefig(filepath)
    plt.cla()
    plt.clf()

def draw(
    cam_pose,
    fov_mask, # 60, 60, 36
    voxels,          # semantic occupancy predictions
    vox_origin=None,
    voxel_size=0.2,  # voxel size in the real world
    sem=False,
    save_path=None
):
    w, h, z = voxels.shape # 60, 60, 36
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask.reshape(-1)]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 12)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=11, # 16
        )
  
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    
    if sem:
        colors = np.array(
            [
                [214,  38, 40, 255],       # "ceiling"             orange
                [43, 160, 4, 255],       # "floor"              pink
                [158, 216, 229, 255],       # "wall"                  yellow
                [  114, 158, 206, 255],       # "window"                  blue
                [  204, 204, 91, 255],       # "chair"  cyan
                [255, 186, 119, 255],       # "bed"           dark orange
                [147, 102, 188, 255],       # "sofa"           red
                [30, 119, 181, 255],       # "table"         light yellow
                [160, 188, 33, 255],       # "tvs"              brown
                [255, 127, 12, 255],       # "furn"                purple                
                [196, 175, 214, 255],       # "objs"   dark pink
            ]
        ).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    mlab.view(
        azimuth=180,
        elevation=0
    )

    if offscreen:
        mlab.savefig(save_path, size=(2560, 1440))
    else:
        mlab.show()
    mlab.close()


def pass_print(*args, **kwargs):
    pass

def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0

def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
        backend="nccl", init_method=f"env://", 
        world_size=world_size, rank=rank
    )

    dist.barrier()
    torch.cuda.set_device(gpu)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='indoor_nyu', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')


    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    
    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)
    my_model.globalhead.requires_grad_(False)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')
    # build dataloader
    from dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()

    CalMeanIou = SSCMetrics(n_classes=12)
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
    
    save_dir_occ = os.path.join(args.work_dir, 'vis_occ_occupancy')
    os.makedirs(save_dir_occ, exist_ok=True)
    save_dir_gauss = os.path.join(args.work_dir, 'vis_occ_gaussian')
    os.makedirs(save_dir_gauss, exist_ok=True)      
    
    scenemeta_keys = ['global_scene_dim', 'global_scene_size', 'global_labels', 'global_pts', 'global_scene_origin', 'global_mask']
    metas_tensor_keys_inv = ['name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'img_shape', 'img_aug_matrix', 'img_depthbranch']
    
    my_model.eval()
    CalMeanIou.reset()
    loss_record = LossRecord(loss_func=loss_func)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    
    with torch.no_grad():
        
        for i_iter, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, labels) = data # imgs [1, 1, 30, 3, 480, 640]  labels [1, 30, 60, 60, 36]
            scenemeta = metas[0]
            for k, v in scenemeta.items():
                if k in scenemeta_keys:
                    scenemeta[k] = torch.tensor(v).cuda()
            K_Frames = len(scenemeta['monometa_list'])
            monometa_list_cuda = []
            for i in range(K_Frames):
                monometa = scenemeta['monometa_list'][i]
                for k, v in monometa.items():
                    if not (k in metas_tensor_keys_inv):
                        monometa[k] = torch.tensor(v).cuda()
                monometa['img_depthbranch'] = monometa['img_depthbranch'].cuda()
                monometa_list_cuda.append(monometa)
            
            my_model.module.scene_init(scenemeta)
            
            save_dir_occ_thisscene = os.path.join(save_dir_occ, scenemeta['scene_name'])
            os.makedirs(save_dir_occ_thisscene, exist_ok=True)
            save_dir_gauss_thisscene = os.path.join(save_dir_gauss, scenemeta['scene_name'])
            os.makedirs(save_dir_gauss_thisscene, exist_ok=True)
            
            for i in range(K_Frames):
                
                img = imgs[:, :, i, :, :, :].unsqueeze(2) # 1, 1, 1, 3, H, W
                label = labels[:, i, :, :, :].unsqueeze(1) # 1, 1, 60, 60, 36
                meta = [monometa_list_cuda[i]]
                
                with torch.cuda.amp.autocast(cfg.amp):
                    result_dict, my_occ, predtoreturn, gaussianstensor_to_return, instance_feature_toreturn, gaussians_to_vis = my_model(scenemeta=scenemeta, imgs=img, metas=meta, points=None, label=label, grad_frames=cfg.grad_frames, test_mode=False)
                my_model.module.scene_update(scenemeta, gaussianstensor_to_return, instance_feature_toreturn, meta[0]['mask_in_global_from_this'])
                
                my_model.module.globalhead.empty_scalar = my_model.module.head.empty_scalar
                my_model.module.globalhead.empty_scale = my_model.module.head.empty_scale
                my_model.module.globalhead.empty_rot = my_model.module.head.empty_rot
                my_model.module.globalhead.empty_sem = my_model.module.head.empty_sem
                my_model.module.globalhead.empty_opa = my_model.module.head.empty_opa
                
                global_gaussians = my_model.module.get_global_gaussian(scenemeta, meta[0]['vox_origin'], meta[0]['scene_size'])
                draw_gaussian(scenemeta['global_scene_origin'].cpu().numpy(), 
                            scenemeta['global_scene_size'].cpu().numpy(), 
                            save_dir_gauss_thisscene, 
                            global_gaussians, idx=0, sem=True,
                            K_idx=i,
                            scene_name=scenemeta['scene_name'])
                
                # begin occ vis
                scene_result_dict = my_model.module.get_global_occ(scenemeta, meta[0]['vox_origin'], meta[0]['scene_size'])
                global_valid_mask = scene_result_dict['mask']
                voxel_label = scene_result_dict['label'].long()
                voxel_predict = scene_result_dict['predict'].long()
                voxel_origin = scenemeta['global_scene_origin'].cpu().numpy()
                resolution = 0.08
                    
                my_mask = (voxel_label==0)
                voxel_label[voxel_label==0] = 12
                to_vis = voxel_label.clone().cpu().numpy()
                
                save_path = os.path.join(save_dir_occ_thisscene, scenemeta['scene_name']+f'_K_{i}_gt.png')
                draw(None,
                    global_valid_mask.clone().cpu().numpy(),
                    to_vis, 
                    voxel_origin, 
                    [resolution] * 3, 
                    sem=True,
                    save_path=save_path)
                
                voxel_predict[my_mask] = 12
                voxel_predict[voxel_predict==0] = 12
                to_vis = voxel_predict.clone().cpu().numpy()
                save_path = os.path.join(save_dir_occ_thisscene, scenemeta['scene_name']+f'_K_{i}_predict.png')
                draw(None,
                    global_valid_mask.clone().cpu().numpy(),
                    to_vis, 
                    voxel_origin, 
                    [resolution] * 3, 
                    sem=True,
                    save_path=save_path)
                # end occ vis
                    
                gc.collect()
                torch.cuda.empty_cache()
                

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/vis_embodied_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/vis_embodied')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=(2560, 1440))
    display.start()
    offscreen = True
# torchrun --nproc_per_node=1 vis_mono.py
from mayavi import mlab
import mayavi
import cv2
import open3d as o3d
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from dataset.nyu_utils import world2pix
from utils.iou_eval import IOUEvalBatch
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
from matplotlib import pyplot as plt, cm, colors
from pyquaternion import Quaternion

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/Depth-Anything-V2/metric_depth')
from train_utils import compute_CP_mega_matrix, downsample_label

def pass_print(*args, **kwargs):
    pass

def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0

def draw(voxel_label, voxel_size=0.05, intrinsic=None, cam_pose=None, d=0.5, save_path=None):
    """Visualize the gt or predicted voxel labels.
    
    Args:
        voxel_label (ndarray): The gt or predicted voxel label, with shape (N, 4), N is for number 
            of voxels, 7 is for [x, y, z, label].
        voxel_size (double): The size of each voxel.
        intrinsic (ndarray): The camera intrinsics.
        cam_pose (ndarray): The camera pose.
        d (double): The depth of camera model visualization.
    """
    figure = mlab.figure(size=(1600*0.8, 900*0.8), bgcolor=(1, 1, 1))
    
    if intrinsic is not None and cam_pose is not None:
        assert d > 0, 'camera model d should > 0'
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # half of the image plane size
        y = d * 2 * cy / (2 * fy)
        x = d * 2 * cx / (2 * fx)
        
        # camera points (cam frame)
        tri_points = np.array(
            [
                [0, 0, 0],
                [x, y, d],
                [-x, y, d],
                [-x, -y, d],
                [x, -y, d],
            ]
        )
        tri_points = np.hstack([tri_points, np.ones((5, 1))])
        
        # camera points (world frame)
        tri_points = (cam_pose @ tri_points.T).T
        x = tri_points[:, 0]
        y = tri_points[:, 1]
        z = tri_points[:, 2]
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]

        # draw cam model
        mlab.triangular_mesh(
            x,
            y,
            z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=7.5,
        )
    # draw occupied voxels
    plt_plot = mlab.points3d(
        voxel_label[:, 0],
        voxel_label[:, 1],
        voxel_label[:, 2],
        voxel_label[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=12,
    )

    # label colors
    colors = np.array(
        [
            [0, 0, 0, 255],  # 0 empty
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
            [128, 128, 128, 255],  # 12 occupied with semantic
        ]
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    
    cam_direction_world = cam_pose[:3, :3] @ np.array([0, 0, 1])
    azimuth = np.arctan2(cam_direction_world[1], cam_direction_world[0]) * 180 / np.pi
    elevation = np.arctan2(cam_direction_world[2], np.linalg.norm(cam_direction_world[:2])) * 180 / np.pi
    mlab.view(azimuth=azimuth, elevation=elevation-22.5)

    mlab.savefig(save_path)
    mlab.close()


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
    logger = MMLogger(name='indoor_nyu_eval', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    
    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)
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
    from dataset import build_dataloader, custom_collate_fn
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
        
    save_dir = os.path.join(args.work_dir, 'vis_occ')
    os.makedirs(save_dir, exist_ok=True)

    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']
    
    my_model.eval()
    loss_record = LossRecord(loss_func=loss_func)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data
            
            for k, v in metas[0].items():
                if not (k in metas_tensor_keys_inv):
                    metas[0][k] = torch.tensor(v).cuda()
            metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()
            
            
            with torch.cuda.amp.autocast(cfg.amp):
                result_dict, my_occ, predtoreturn = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=None, test_mode=True)
            
            voxel_predict = torch.argmax(result_dict['ce_input'], dim=1).long()
            voxel_label = result_dict['ce_label'].long()
            voxel_origin = metas[0]['vox_origin'].cpu().numpy()
            resolution = 0.08
            cam_pose = metas[0]['cam2world'].cpu().numpy()
            cam_k = metas[0]['cam_k'].cpu().numpy()
            
            for i in range(voxel_label.shape[0]):
                
                my_mask = (voxel_label[i]==0)
                voxel_label[i][voxel_label[i]==12] = 0
                to_vis = voxel_label[i].reshape(-1)
                to_vis_xyz = metas[0]['occ_xyz'].reshape(-1, 3)
                mask1 = (to_vis == 0)
                fov_mask = result_dict['fov_mask'].reshape(-1)
                mask = (~mask1) & fov_mask
                to_vis = to_vis[mask]
                to_vis_xyz = to_vis_xyz[mask]
                to_vis = torch.cat([to_vis_xyz, to_vis.unsqueeze(-1)], dim=-1)
                to_vis = to_vis.cpu().numpy()
                
                save_path = os.path.join(save_dir, metas[0]['name'].replace('/', '')+'gt.png')
                draw(to_vis, voxel_size=0.08, intrinsic=cam_k, cam_pose=cam_pose, d=0.5,
                           save_path=save_path)
                
                voxel_predict[i][my_mask] = 12
                voxel_predict[i][voxel_predict[i]==12] = 0
                to_vis = voxel_predict[i].reshape(-1)
                to_vis_xyz = metas[0]['occ_xyz'].reshape(-1, 3)
                mask2 = (to_vis == 0)
                fov_mask = result_dict['fov_mask'].reshape(-1)
                mask = (~mask2) & fov_mask
                to_vis = to_vis[mask]
                to_vis_xyz = to_vis_xyz[mask]
                to_vis = torch.cat([to_vis_xyz, to_vis.unsqueeze(-1)], dim=-1)
                to_vis = to_vis.cpu().numpy()
                
                save_path = os.path.join(save_dir, metas[0]['name'].replace('/', '')+'predict.png')
                draw(to_vis, voxel_size=0.08, intrinsic=cam_k, cam_pose=cam_pose, d=0.5,
                           save_path=save_path)

    

        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/vis_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/vis_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0])

    args, _ = parser.parse_known_args()
    main(args)
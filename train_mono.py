import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torchrun --nproc_per_node=8 train_mono.py
from utils.iou_eval import IOUEvalBatch
from utils.iou_as_iso import SSCMetrics
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/gl/zph_ws/EmbodiedOcc')
sys.path.append('/home/gl/zph_ws/EmbodiedOcc/Depth-Anything-V2/metric_depth')
from PIL import Image

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

    # dist.barrier()
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

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=1000, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    CalMeanIou = SSCMetrics(n_classes=12)
    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

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
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
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
        
    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']
    
    # training
    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data
            for k, v in metas[0].items():
                if not (k in metas_tensor_keys_inv):
                    metas[0][k] = torch.tensor(v).cuda()
            metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            
            with torch.cuda.amp.autocast(enabled=amp):
                result_dict, my_occ, predtoreturn = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=cfg.grad_frames, test_mode=False)
            
            loss, loss_dict = loss_func(result_dict)
            loss_record.update(loss=loss.item(), loss_dict=loss_dict)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

            valid_grad = True
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_iter)
            time_e = time.time()
            if not valid_grad and is_main_process():
                logger.info('[Nan Grad] Epoch %d Iter %5d' % (epoch+1, i_iter))
                params, grads = [], []
                for name, param in my_model.named_parameters():
                    if param.requires_grad:
                        params.append(param.abs().mean().item())
                        grads.append(param.grad.abs().mean().item())
                logger.info('%.5f     %.5f     %.5f' % (loss.item(), torch.mean(torch.tensor(params)).item(), torch.mean(torch.tensor(grads)).item()))

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()
            
            gc.collect()
            torch.cuda.empty_cache()
            
        # save checkpoint
        if is_main_process():
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_iou': best_val_iou,
                'best_val_miou': best_val_miou
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
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
                    
                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, my_occ, predtoreturn = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=None, test_mode=True)
                    
                    loss, loss_dict = loss_func(result_dict)
                    loss_record.update(loss=loss.item(), loss_dict=loss_dict)
                    
                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                    voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]
                    
                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0
                    voxel_predict = voxel_predict.cpu()
                    voxel_label = voxel_label.cpu()
                    
                    CalMeanIou.add_batch(voxel_predict, voxel_label)
                                        
                    if i_iter_val % print_freq == 0 and is_main_process():
                        loss_info = loss_record.loss_info()
                        logger.info('[EVAL] Iter %5d/%d   '%(i_iter_val, len(val_dataset_loader)) + loss_info)
                        
                    gc.collect()
                    torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats()
            
            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]
            
            logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current val iou of sem is {info_sem}')
            logger.info(f'Current val iou of geo is {info_geo}')
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/gl/zph_ws/EmbodiedOcc/workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
    
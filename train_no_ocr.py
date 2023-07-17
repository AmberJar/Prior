import os
import shutil
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
import models
import data
import numpy as np
import time
from tqdm import tqdm
from torch.utils import tensorboard
from helper.metrics import eval_metrics
from utils.recorder import Recorder
from utils.get_params import get_hparams
from torch.cuda.amp import autocast, GradScaler
from helper.scheduler import WarmUpLR_CosineAnnealing
from helper.loss_helper import LSCE_GDLoss, LabelSmoothSoftmaxCE
from torch.nn.parallel import DistributedDataParallel as DDP
from data.cityscapes_mask_dataloader import CityScapes
from data.general_dataloder import General
from data.hires_dataset import HiResNetDataLoader
import datetime

from LibMTL.weighting.RLW import RLW


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, getattr(config, name).type)(*args, **getattr(config, name).args)  # getattar(类名，属性）（方法）


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
             args=(world_size, config,),
             nprocs=world_size,
             join=True)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def main(rank, world_size, config):
    seed_everything(seed=88)
    setup(rank, world_size)
    cfg = get_hparams(config)

    start_time = str(datetime.datetime.fromtimestamp(int(time.time())))
    start_time = start_time.replace(' ', '@')

    num_classes = cfg.num_classes

    # 创建checkpoint路径
    net_path = os.path.join(cfg.trainer.save_dir, cfg.name)
    if not os.path.exists(net_path):
        os.mkdir(net_path)

    # 加载模型，简易版
    model = models.HiResNetWithoutOCR(num_classes=num_classes, backbone='hrnet48').to(rank)

    # 计算参数量
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .2fM' % (total / 1e6))

    model = DDP(model, device_ids=[rank])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 加载数据集
    args = {
        'data_dir': cfg.train_loader.data_dir,
        'batch_size': cfg.batch_size,
        'num_classes': num_classes,
    }

    train_loader = HiResNetDataLoader(data_dir=cfg.train_loader.data_dir,
                                      batch_size=cfg.batch_size,
                                      split=cfg.train_loader.split,
                                      num_workers=cfg.train_loader.num_workers,
                                      num_classes=num_classes,
                                      mosaic_ratio=0.25,
                                      mode='random_mask',
                                      augment=False)

    val_loader = HiResNetDataLoader(data_dir=cfg.val_loader.data_dir,
                                    batch_size=cfg.batch_size*4,
                                    split=cfg.val_loader.split,
                                    num_workers=cfg.val_loader.num_workers,
                                    num_classes=num_classes,
                                    mode='random_mask',
                                    augment=False)

    # 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.optimizer.lr,
                            weight_decay=cfg.optimizer.weight_decay,
                            betas=cfg.optimizer.betas)

    lr_scheduler = WarmUpLR_CosineAnnealing(optimizer=optimizer,
                                            num_epochs=cfg.trainer.epochs,
                                            iters_per_epoch=len(train_loader),
                                            warmup_epochs=cfg.lr_scheduler.warmup_epochs)

    writer = None

    if rank == 0:
        start_times = datetime.datetime.now().strftime('%m-%d_%H-%M')
        writer_dir = os.path.join(cfg.trainer.log_dir, cfg.name, start_times)

        writer_dir = writer_dir.replace('\\', '/')
        # shutil.copy(config, writer_dir)
        writer = tensorboard.SummaryWriter(writer_dir)
    scaler = GradScaler(enabled=cfg.trainer.fp16)
    loss_fn = LSCE_GDLoss(ignore_index=255)

    for epoch in range(0, cfg.trainer.epochs):
        train_epoch(rank, cfg, train_loader, model, optimizer, loss_fn, scaler, epoch, num_classes, writer)
        lr_scheduler.step()

        if epoch % cfg.trainer.val_per_epochs == 0:
            evaluate(rank, cfg, model, val_loader, loss_fn, epoch, num_classes, writer)

        if rank == 0:
            if epoch % cfg.trainer.save_period == 0:
                checkpoint_dir = os.path.join(cfg.trainer.save_dir, cfg.name, start_time)
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                save_checkpoint(epoch, model, optimizer, cfg, checkpoint_dir, save_best=False)


def train_epoch(rank, cfg, loaders, model, optimizer, loss_fn, scaler, epoch, num_classes, writer):
    train_loader = loaders
    model.train()
    wrt_mode, wrt_step = 'train', 0
    log_step = cfg.trainer.log_per_iter
    num_classes = cfg.num_classes
    alpha = cfg.alpha
    # 进度条设置
    if rank == 0:
        tbar = tqdm(train_loader, ncols=130)
    else:
        tbar = train_loader

    # 数据记录初始化
    recorder = Recorder()
    recorder.reset_metrics()
    RLW_fn = RLW(device=rank)
    for index, (images, labels) in enumerate(tbar):
        images = images.to(rank)
        labels = labels.to(rank)

        optimizer.zero_grad()

        if cfg.trainer.fp16:
            r = np.random.rand(1)

            # beta，迪利克雷分布参数，正反硬币次数，整数
            # cutmix_prob，cutmix的概率
            cutmix_prob = 0.5
            beta = 1

            if cfg.trainer.cutmix:
                if beta > 0 and r < cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(images.size()[0]).cuda()
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

                    with autocast(enabled=cfg.trainer.fp16):
                        # out_aux, output = model(images)
                        output = model(images)
                        # compute output
                        loss = loss_fn(output, target_a) * lam + loss_fn(output, target_b) * (1. - lam)
                        # loss_aux = loss_fn(out_aux, target_a) * lam + loss_fn(out_aux, target_b) * (1. - lam)
                        # loss = alpha * loss_aux + loss
                else:
                    # compute output
                    with autocast(enabled=cfg.trainer.fp16):
                        # out_aux, output = model(images)
                        output = model(images)
                        # loss_aux = loss_fn(out_aux, labels)
                        loss = loss_fn(output, labels)
                        # loss = alpha * loss_aux + loss
            else:
                with autocast(enabled=cfg.trainer.fp16):
                    # out_aux, output = model(images)
                    output = model(images)
                    loss_aux = loss_fn(out_aux, labels)
                    loss = loss_fn(output, labels)
                    # loss = alpha * loss_aux + loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            r = np.random.rand(1)

            # beta，迪利克雷分布参数，正反硬币次数，整数
            # cutmix_prob，cutmix的概率
            cutmix_prob = 0.5
            beta = 1

            if beta > 0 and r < cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                out_aux, output = model(images)
                # output = model(images)
                # compute output
                loss = loss_fn(output, target_a) * lam + loss_fn(output, target_b) * (1. - lam)
                loss_aux = loss_fn(out_aux, target_a) * lam + loss_fn(out_aux, target_b) * (1. - lam)
            else:
                # compute output
                out_aux, output = model(images)
                # output = model(images)
                loss = loss_fn(output, labels)
                loss_aux = loss_fn(out_aux, labels)

            loss = alpha * loss_aux + loss

            loss.backward()
            optimizer.step()

        # 记录loss
        recorder.total_loss.update(loss.item())

        # 计算各项指标并上传
        seg_metrics = eval_metrics(output, labels, num_classes)
        recorder.update_seg_metrics(*seg_metrics)
        pixAcc, mIoU, Class_IoU = recorder.get_seg_metrics(num_classes).values()

        if rank == 0:
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.3f} | Acc {:.3f} mIoU {:.3f}'.format(
                    epoch, recorder.total_loss.average,
                    pixAcc, mIoU))
            if index % log_step == 0:
                wrt_step = epoch * len(train_loader) + index
                writer.add_scalar(f'{wrt_mode}/loss', loss.item(), wrt_step)

    seg_metrics = recorder.get_seg_metrics(num_classes)
    if rank == 0:
        for k, v in list(seg_metrics.items())[:-1]:
            writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
        for i, opt_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], wrt_step)

    log = {
        'loss': recorder.total_loss.average,
        **seg_metrics
    }
    return log


def evaluate(rank, cfg, model, val_loader, loss_fn, epoch, num_classes, writer):
    model.eval()

    tbar = tqdm(val_loader, ncols=130)
    wrt_mode = 'val'

    # 数据记录初始化
    recorder = Recorder()
    recorder.reset_metrics()

    with torch.no_grad():
        for index, (images, labels) in enumerate(tbar):
            images = images.to(rank)
            labels = labels.to(rank)

            # out_aux, outs = model(images)
            outs = model(images)
            # loss_aux = loss_fn(out_aux, labels)
            loss = loss_fn(outs, labels)
            # loss = loss_aux + loss

            # 记录loss
            recorder.total_loss.update(loss.item())

            # 计算各项指标并上传
            seg_metrics = eval_metrics(outs, labels, num_classes)
            recorder.update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, Class_IoU = recorder.get_seg_metrics(num_classes).values()

            if rank == 0:
                tbar.set_description(
                    'VAL ({}) | Loss: {:.3f} | Acc {:.3f} mIoU {:.3f}'.format(
                        epoch, recorder.total_loss.average,
                        pixAcc, mIoU))

        seg_metrics = recorder.get_seg_metrics(num_classes)
        if rank == 0:
            wrt_step = epoch * len(val_loader)
            writer.add_scalar(f'{wrt_mode}/loss', recorder.total_loss.average, wrt_step)

            # 这里的writer的记录直接写就行了
            for k, v in list(seg_metrics.items())[:-1]:
                writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)

        log = {
            'loss': recorder.total_loss.average,
            **seg_metrics
        }

    model.train()


def save_checkpoint(epoch, model, optimizer, cfg, checkpoint_dir, save_best=False):
    state = {
        'arch': type(model).__name__,
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': cfg
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    # logger.info(f'\nSaving a checkpoint: {filename} ...')
    torch.save(state, filename)

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(state, filename)
        # logger.info("Saving current best: best_model.pth")


def start():
    config = './config/config_hires.json'
    n_gpus = torch.cuda.device_count()
    run_demo(main, world_size=n_gpus, config=config)


if __name__ == '__main__':
    start()

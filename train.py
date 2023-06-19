import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import random
import models
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
import datetime


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
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])  # getattar(类名，属性）（方法）


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


def main(rank, world_size, config):
    setup(rank, world_size)
    cfg = get_hparams(config)
    start_time = time.asctime(time.localtime())

    num_classes = cfg.num_classes

    # 创建checkpoint路径
    net_path = os.path.join(cfg.trainer.save_dir, cfg.name)
    if not os.path.exists(net_path):
        os.mkdir(net_path)

    # 加载模型，简易版
    model = models.PriorNet(num_classes=num_classes).to(rank)
    model = DDP(model, device_ids=[rank])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 加载数据集
    train_loader = General(data_dir=cfg.train_loader.data_dir,
                           batch_size=cfg.batch_size,
                           split=cfg.train_loader.split,
                           num_workers=cfg.train_loader.num_workers,
                           num_classes=num_classes,
                           mode='random_mask',
                           augment=False)

    val_loader = General(data_dir=cfg.val_loader.data_dir,
                         batch_size=cfg.batch_size,
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

        writer = tensorboard.SummaryWriter(writer_dir)
    scaler = GradScaler(enabled=cfg.trainer.fp16)
    loss_fn = LSCE_GDLoss(ignore_index=255)

    for epoch in range(1, cfg.trainer.epochs):
        train_epoch(rank, cfg, train_loader, model, optimizer, loss_fn, scaler, epoch, num_classes, writer)
        lr_scheduler.step()
        if rank == 0:
            if epoch % cfg.trainer.val_per_epochs == 0:
                evaluate(rank, cfg, model, val_loader, loss_fn, epoch, num_classes, writer)

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

    # 进度条设置
    if rank == 0:
        tbar = tqdm(train_loader, ncols=130)
    else:
        tbar = train_loader

    # 数据记录初始化
    recorder = Recorder()
    recorder.reset_metrics()

    for index, (images, labels, masks) in enumerate(tbar):
        images = images.to(rank)
        labels = labels.to(rank)
        masks = masks.to(rank)

        with autocast(enabled=cfg.trainer.fp16):
            outs = model((images, masks))
            loss = loss_fn(outs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # 记录loss
        recorder.total_loss.update(loss.item())

        # 计算各项指标并上传
        seg_metrics = eval_metrics(outs, labels, num_classes)
        recorder.update_seg_metrics(*seg_metrics)
        pixAcc, mIoU, Class_IoU = recorder.get_seg_metrics(num_classes).values()

        if rank == 0:
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f}'.format(
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
        for index, (images, labels, masks) in enumerate(tbar):
            images = images.to(rank)
            labels = labels.to(rank)
            masks = masks.to(rank)

            outs = model.infernce((images, masks))
            loss = loss_fn(outs, labels)

            # 记录loss
            recorder.total_loss.update(loss.item())

            # 计算各项指标并上传
            seg_metrics = eval_metrics(outs, labels, num_classes)
            recorder.update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, Class_IoU = recorder.get_seg_metrics(num_classes).values()

            if rank == 0:
                tbar.set_description(
                    'VAL ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f}'.format(
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
    config = './config/config_init.json'
    n_gpus = torch.cuda.device_count()
    run_demo(main, world_size=n_gpus, config=config)


if __name__ == '__main__':
    start()

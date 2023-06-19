import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from tqdm import tqdm
from collections import OrderedDict
import argparse
from utils.get_params import get_hparams
from models.hrnet import HRNet_W48_OCR
from data.general_dataloder import General

from helper.metrics import eval_metrics
from utils.recorder import Recorder


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='./config/config_init.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model',
                        default='./pretrained/hrnet_w48_ocr_1_latest.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    args = parser.parse_args()
    return args


def predicts(model, device, cfg):
    cfg.val_loader.data_dir = '/data/fangpengcheng/data/Cityscapes'
    cfg.val_loader.split = 'test'

    ValLoader = General(data_dir=cfg.val_loader.data_dir,
                        batch_size=cfg.batch_size,
                        split=cfg.val_loader.split,
                        num_workers=cfg.val_loader.num_workers,
                        num_classes=cfg.num_classes,
                        mode='random_mask',
                        augment=False)

    tbar = tqdm(ValLoader, ncols=130)

    # 数据记录初始化
    recorder = Recorder()
    recorder.reset_metrics()

    with torch.no_grad():
        for index, (images, labels, _) in enumerate(tbar):
            images = images.to(device)
            labels = labels.to(device)

            outs, _ = model(images)

            # 计算各项指标并上传
            seg_metrics = eval_metrics(outs, labels, cfg.num_classes)
            recorder.update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, Class_IoU = recorder.get_seg_metrics(cfg.num_classes).values()

            tbar.set_description(
                'Testing | Acc {:.2f} mIoU {:.2f}'.format(pixAcc, mIoU))

        seg_metrics = recorder.get_seg_metrics(cfg.num_classes)
        print(seg_metrics)


def predict():
    args = parse_arguments()
    config = './config/config_init.json'
    cfg = get_hparams(config)

    # Model
    print("Load model ............")
    model = HRNet_W48_OCR(num_classes=cfg.num_classes, backbone="hrnet48")
    available_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda' if len(available_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model)
    checkpoint = checkpoint['state_dict']

    if 'module' in list(checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict

    # load
    model.to(device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("Load model complete.>>>")

    predicts(model, device, cfg)


if __name__ == '__main__':
    predict()

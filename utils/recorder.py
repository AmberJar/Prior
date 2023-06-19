from helper.metrics import eval_metrics, AverageMeter
import numpy as np

class Recorder:
    def __init__(self):
        self.total_label = None
        self.total_union = None
        self.total_correct = None
        self.total_inter = None
        self.data_time = None
        self.total_loss = None
        self.batch_time = None

    def reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get_seg_metrics(self, num_classes):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(IoU[:-1].mean(), 3),
            "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
        }



if __name__ == '__main__':
    recorder = Recorder()

    # 调用
    recorder.reset_metrics()
    loss = 1

    recorder.total_loss.update(loss)

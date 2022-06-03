import os
import visdom
import numpy as np
import test_time
import cv2
import math
from sklearn import metrics

def mkdir(path):
    # 引入模块
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1.0 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 定义获取当前学习率的函数
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        # 画的第几个数，相当于横坐标
        # 比如("loss", 23) 即loss的第23个点
        self.index = {}
        self.log_text = ""

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env

        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ("loss", 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        # self.plot("loss", 1.00)

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else "append",
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img("input_img", t.Tensor(64, 64))
        self.img("input_imgs", t.Tensor(3, 64, 64))
        self.img("input_imgs", t.Tensor(100, 1, 64, 64))
        self.img("input_imgs", t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win="log_text"):
        """
        self.log({"loss": 1, "lr": 0.0001})
        """
        self.log_text += ("[{time}] {info} <br>".format(
            time=test_time.strftime("%m%d_%H%M%S"), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot, image, log, plot_many等除外
        """
        return getattr(self.vis, name)


def max_fusion(x, y):
    assert x.shape == y.shape

    return np.maximum(x, y)


def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()

    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])

        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)

    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec, pred_vec)

    return roc_auc


def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)

    FP = np.float32(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float32(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float32(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float32(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))

    return FP, FN, TP, TN


def calc_acc(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)

    return acc


def calc_sen(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    sen = TP / (FN + TP + 1e-12)

    return sen


def calc_fdr(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    fdr = FP / (FP + TP + 1e-12)

    return fdr


def calc_spe(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    spe = TN / (FP + TN + 1e-12)

    return spe


def calc_gmean(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    sen = calc_sen(pred_arr, gt_arr, kernel_size=kernel_size)
    spe = calc_spe(pred_arr, gt_arr, kernel_size=kernel_size)

    return math.sqrt(sen * spe)


def calc_kappa(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size=kernel_size)
    matrix = np.array([[TP, FP],
                       [FN, TN]])
    n = np.sum(matrix)

    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col

    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)

    return (po - pe) / (1 - pe)


def calc_iou(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    iou = TP / (FP + FN + TP + 1e-12)

    return iou


def calc_dice(pred_arr, gt_arr, kernel_size=(3, 3)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)

    return dice

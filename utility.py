import time
import random
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import scipy.io as sio
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import math
import glob
import re
from os.path import *

def optset(opt):
    if opt.dataset == 'CAVE':
        opt.n_colors = 10
        opt.channel = 29
        opt.hschannel = 29
        opt.mschannel = 3
        opt.trainset_sizeIX = 512
        opt.trainset_sizeIY = 512
        opt.trainset_file_num = 20
        opt.testset_sizeI = 512
        opt.testset_file_num = 12
    elif opt.dataset == 'Harvard':
        opt.n_colors = 8
        opt.channel = 31
        opt.hschannel = 31
        opt.mschannel = 3
        opt.trainset_sizeIX = 1024
        opt.trainset_sizeIY = 1024
        opt.trainset_file_num = 30
        opt.testset_sizeI = 1024
        opt.testset_file_num = 20
    elif opt.dataset == 'Pavia':
        opt.n_colors = 20
        opt.channel = 102
        opt.hschannel = 102
        opt.mschannel = 4
        opt.trainset_sizeIX = 1024
        opt.trainset_sizeIY = 459
        opt.trainset_file_num = 1
        opt.testset_sizeI = 256
        opt.testset_file_num = 4
    elif opt.dataset == 'Chikusei':
        opt.n_colors = 20
        opt.channel = 128
        opt.hschannel = 128
        opt.mschannel = 3
        opt.trainset_sizeIX = 512
        opt.trainset_sizeIY = 512
        opt.trainset_file_num = 10
        opt.testset_sizeI = 512
        opt.testset_file_num = 6
    else:
        raise ValueError(f"Invalid dataset type: {opt.dataset}.")
    return opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret


def PSNR(reference, target, data_range):
    bands = reference.shape[2]
    mpsnr = 0
    for i in range(bands):
        mpsnr += compare_psnr(reference[:, :, i], target[:, :, i], data_range=data_range)
    mpsnr /= bands
    return mpsnr


def SSIM(reference, target, data_range):
    '''
    平均结构相似性
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    _, _, bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += compare_ssim(reference[:,:,i],target[:,:,i],data_range=data_range)
    mssim /= bands
    return mssim


def SAM(reference, target):
    '''
    光谱角度映射器评价（求取平均光谱映射角度，理想值为0）
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)
    prod_scal = dot(reference, target)
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:0]
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    angolo = np.sum(np.arccos(prod_scal / prod_norm)) / prod_scal.shape[0]
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam



def ERGAS(references, target, ratio):
    rows, cols, bands = references.shape
    d = 1 / ratio
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels
    uk = np.mean(tar_temp, axis=0)
    relative_rmse2 = rmse2 / uk ** 2
    total_relative_rmse = np.sum(relative_rmse2)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)
    return out


def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    return optimizer_function(trainable, **kwargs)


def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.epochs),
        eta_min=opt.eta_min
    )
    return scheduler


def dot(m1, m2):
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def Upsample(sr, MSI, HSI, sf):
    data = sio.loadmat('data/CAVE/response coefficient')
    R = data['R'].T
    B = data['B512']

    mu = 1e-3
    nr = MSI.shape[0]
    nc = MSI.shape[1]
    L = HSI.shape[2]
    HSI_int = np.zeros((nr, nc, L))
    HSI_int[0::sf, 0::sf, :] = HSI
    FBmC = np.conj(B)
    FBs = np.tile(B[:, :, np.newaxis], (1, 1, L))
    FBCs1 = np.tile(FBmC[:, :, np.newaxis], (1, 1, L))
    HHH = ifft2((fft2(HSI_int) * FBCs1))
    HHH1 = hyperConvert2D(HHH)
    MSI3 = reshape(np.moveaxis(MSI, 2, 0), (MSI.shape[2], -1))
    n_dr = nr // sf
    n_dc = nc // sf
    V2 = sr
    CCC = np.dot(R.T, MSI3) + HHH1
    C1 = np.dot(R.T, R) + mu * np.eye(R.shape[1])
    Lambda, Q = np.linalg.eig(C1)
    Lambda = reshape(Lambda, (1, 1, L))
    InvLbd = 1 / np.tile(Lambda, (sf * n_dr,  sf * n_dc, 1))
    B2Sum = PPlus(np.power(np.abs(FBs), 2) / (sf * sf), n_dr, n_dc)
    InvDI = 1 / (B2Sum[0:n_dr, 0:n_dc, :] + np.tile(Lambda, (n_dr, n_dc, 1)))
    HR_HSI3 = mu * V2
    C3 = CCC + HR_HSI3
    C30 = fft2(reshape(np.dot(np.linalg.inv(Q), C3).T, (nr, nc, L))) * InvLbd
    temp = PPlus_s(C30 / (sf * sf) * FBs, n_dr, n_dc)
    invQUF = C30 - np.tile(temp * InvDI, (sf, sf, 1)) * FBCs1
    VXF = np.dot(Q, reshape(invQUF, (nr * nc, L)).T)
    ZE = reshape(np.real(ifft2(reshape(VXF.T, (nr, nc, L)))), (nr * nc, L)).T
    Zt = reshape(ZE.T, (nr, nc, -1))
    return Zt


def hyperConvert2D(Image3D):
    h = Image3D.shape[0]
    w = Image3D.shape[1]
    numBands = Image3D.shape[2]
    Image2D = reshape(Image3D, (w * h, numBands)).T
    return Image2D


def PPlus(X, n_dr, n_dc):
    nr = X.shape[0]
    nc = X.shape[1]
    nb = X.shape[2]
    Temp = reshape(X, (nr * n_dc, nc // n_dc, nb))
    Temp[:, 0, :] = np.sum(Temp, 1)
    Temp1 = reshape(np.transpose(reshape(Temp[:, 0, :], (nr, n_dc, nb)), (1, 0, 2)), (n_dc * n_dr, nr // n_dr, nb))
    X[0:n_dr, 0:n_dc, :] = np.transpose(reshape(np.sum(Temp1, 1), (n_dc, n_dr, nb)), (1, 0, 2))
    return X


def PPlus_s(X, n_dr, n_dc):
    nr = X.shape[0]
    nc = X.shape[1]
    nb = X.shape[2]
    Temp = reshape(X, (nr * n_dc, nc // n_dc, nb))
    Temp[:, 0, :] = np.sum(Temp, 1)
    Temp1 = reshape(np.transpose(reshape(Temp[:, 0, :], (nr, n_dc, nb)), (1, 0, 2)), (n_dc * n_dr, nr // n_dr, nb))
    Y = np.transpose(reshape(np.sum(Temp1, 1), (n_dc, n_dr, nb)), (1, 0, 2))
    return Y


def fft2(x):
    return np.fft.fft2(x, axes=(0, 1))


def ifft2(x):
    return np.fft.ifft2(x, axes=(0, 1))


def reshape(x, axes):
    return np.reshape(x, axes, order="F")


def findLastCheckpoint(save_dir):
    file_list = glob.glob(join(save_dir, 'model_last_*.pth'))
    if file_list:
        epochs_exist = []
        for file in file_list:
            result = re.findall("model_last_(\d+)\.pth", file)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch



def findBestCheckpoint(save_dir):
    file_list = glob.glob(join(save_dir, f'model_best_*.pth'))
    if file_list:
        epochs_exist = []
        psnr_exist = []
        for file in file_list:
            result_epochs = re.findall(f'model_best_(.*?)\(.*\.pth', file)
            result_psnr = re.findall(f'model_best_.*?\((.*?)\)\.pth', file)
            epochs_exist.append(int(result_epochs[0]))
            psnr_exist.append(float(result_psnr[0]))
        best_epoch = max(epochs_exist)
        best_psnr = max(psnr_exist)
    else:
        best_epoch = 0
        best_psnr = 0
    return best_epoch, best_psnr


def setInitialBestEpoch(save_dir, initial_epoch):
    best_epoch = 0
    best_psnr = 0
    if initial_epoch > 0:
        best_epoch, best_psnr = findBestCheckpoint(save_dir)
    return best_epoch, best_psnr


def deletePreviousModel(save_dir, name, epoch):
    file_list = glob.glob(join(save_dir, f'{name}_*.pth'))
    if file_list:
        for file in file_list:
            result_epochs = re.findall(f'{name}_(\d+).*\.pth', file)
            if result_epochs and epoch > int(result_epochs[0]):
                os.remove(file)


def preWarmLoader(args, trainset):
    pre_warm_loader = DataLoader(trainset,
        num_workers=args.n_threads,
        pin_memory=True,
        persistent_workers=True)
    it = iter(pre_warm_loader)
    batch = next(it)
    _ = [X.cuda(non_blocking=True) for X in batch[:4]]
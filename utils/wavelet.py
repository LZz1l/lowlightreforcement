import numpy as np
import pywt
import torch


def wavelet_decompose(x, wavelet='db4', level=3):
    """
    对批处理图像进行小波分解
    x: Tensor [B, C, H, W]，范围[0,1]
    返回：低频分量 + 高频分量列表
    """
    x_np = x.detach().cpu().numpy()  # 转为numpy
    b, c, h, w = x_np.shape
    ll_list = []  # 低频分量
    hh_list = []  # 高频分量（(lh, hl, hh)）

    for i in range(b):
        for j in range(c):
            # 小波分解
            coeffs = pywt.wavedec2(x_np[i, j], wavelet=wavelet, level=level)
            ll = coeffs[0]  # 低频
            hh = coeffs[1:]  # 高频
            ll_list.append(ll)
            hh_list.append(hh)

    # 转为Tensor并恢复批维度
    ll_tensor = torch.from_numpy(np.stack(ll_list).reshape(b, c, ll.shape[0], ll.shape[1])).to(x.device)
    return ll_tensor, hh_list


def wavelet_reconstruct(ll, hh_list, wavelet='db4', level=3):
    """小波重构"""
    ll_np = ll.detach().cpu().numpy()
    b, c, h, w = ll_np.shape
    recon_list = []

    for i in range(b):
        for j in range(c):
            # 组合低频和高频分量
            coeffs = [ll_np[i, j]] + hh_list[i * c + j]
            # 重构
            recon = pywt.waverec2(coeffs, wavelet=wavelet)
            recon_list.append(recon)

    # 转为Tensor并裁剪到原尺寸（避免小波分解/重构导致的尺寸偏差）
    recon_tensor = torch.from_numpy(np.stack(recon_list).reshape(b, c, h * 2, w * 2)).to(ll.device)
    return recon_tensor

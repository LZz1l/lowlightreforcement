import numpy as np
import random
import cv2


def random_crop(lq_img, gt_img, crop_size):
    """随机裁剪低光和正常光图像（确保尺寸一致）"""
    h, w = lq_img.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"图像尺寸({h}x{w})小于裁剪尺寸({crop_size}x{crop_size})")

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        lq_img[top:top + crop_size, left:left + crop_size, ...],
        gt_img[top:top + crop_size, left:left + crop_size, ...]
    )


def random_flip(lq_img, gt_img, prob=0.5):
    """随机水平/垂直翻转"""
    if random.random() < prob:  # 水平翻转
        lq_img = lq_img[:, ::-1, ...]
        gt_img = gt_img[:, ::-1, ...]
    if random.random() < prob:  # 垂直翻转
        lq_img = lq_img[::-1, :, ...]
        gt_img = gt_img[::-1, :, ...]
    return lq_img, gt_img


def random_rot90(lq_img, gt_img, prob=0.5):
    """随机旋转90度倍数"""
    if random.random() < prob:
        k = random.randint(1, 3)  # 90/180/270度
        lq_img = np.rot90(lq_img, k=k)
        gt_img = np.rot90(gt_img, k=k)
    return lq_img, gt_img


def image_mixing(lq_img, gt_img):
    """
    图像混合增强（随机选择一种方式）：
    1. 阿尔法混合：加权融合图像与自身翻转版本
    2. 通道洗牌：随机交换RGB通道
    3. 补丁混合：随机替换部分区域为翻转后的补丁
    """
    mix_type = random.choice(['alpha_blend', 'channel_shuffle', 'patch_mix'])
    h, w = lq_img.shape[:2]

    if mix_type == 'alpha_blend':
        # 阿尔法加权融合
        alpha = random.uniform(0.2, 0.8)
        lq_mix = lq_img[:, ::-1, ...]  # 水平翻转作为混合图像
        gt_mix = gt_img[:, ::-1, ...]
        lq_img = alpha * lq_img + (1 - alpha) * lq_mix
        gt_img = alpha * gt_img + (1 - alpha) * gt_mix

    elif mix_type == 'channel_shuffle' and lq_img.shape[-1] == 3:
        # 随机交换RGB通道
        channels = [0, 1, 2]
        random.shuffle(channels)
        lq_img = lq_img[..., channels]
        gt_img = gt_img[..., channels]

    elif mix_type == 'patch_mix':
        # 随机替换部分区域
        patch_h = random.randint(h // 4, h // 2)
        patch_w = random.randint(w // 4, w // 2)
        top = random.randint(0, h - patch_h)
        left = random.randint(0, w - patch_w)
        # 用翻转后的图像补丁替换
        lq_img[top:top + patch_h, left:left + patch_w, ...] = lq_img[::-1, ::-1, ...][top:top + patch_h,
                                                              left:left + patch_w, ...]
        gt_img[top:top + patch_h, left:left + patch_w, ...] = gt_img[::-1, ::-1, ...][top:top + patch_h,
                                                              left:left + patch_w, ...]

    # 确保像素值在[0,1]范围
    return np.clip(lq_img, 0, 1), np.clip(gt_img, 0, 1)
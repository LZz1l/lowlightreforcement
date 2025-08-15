import numpy as np
import random
import cv2


def random_crop(lq_img, gt_img, crop_size):
    """随机裁剪低光和正常光图像（自动适配小尺寸图像）"""
    h, w = lq_img.shape[:2]
    # 确保裁剪尺寸不超过图像尺寸
    crop_size = min(crop_size, h, w)
    if crop_size <= 0:
        raise ValueError(f"无效的裁剪尺寸: {crop_size}（图像尺寸: {h}x{w}）")

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        lq_img[top:top + crop_size, left:left + crop_size, ...],
        gt_img[top:top + crop_size, left:left + crop_size, ...]
    )


def random_flip(lq_img, gt_img, prob=0.5):
    """随机水平/垂直翻转（确保操作一致性）"""
    # 水平翻转
    if random.random() < prob:
            # 水平翻转后立即复制
            lq_img = lq_img[:, ::-1, ...].copy()  # 关键：添加.copy()
            gt_img = gt_img[:, ::-1, ...].copy()
    if random.random() < prob:
            # 垂直翻转后立即复制
            lq_img = lq_img[::-1, :, ...].copy()  # 关键：添加.copy()
            gt_img = gt_img[::-1, :, ...].copy()
    return lq_img, gt_img


def random_rot90(lq_img, gt_img, prob=0.5):
    """随机旋转90度倍数（确保维度一致性）"""
    if random.random() < prob:
        k = random.randint(1, 3)  # 90/180/270度
        lq_img = np.rot90(lq_img, k=k, axes=(0, 1)).copy()  # 明确指定旋转轴（H,W）
        gt_img = np.rot90(gt_img, k=k, axes=(0, 1)).copy()
    return lq_img, gt_img


def image_mixing(lq_img, gt_img):
    """图像混合增强（增加通道检查）"""
    # 确保输入是float32且在[0,1]范围
    lq_img = np.clip(lq_img.astype(np.float32), 0, 1)
    gt_img = np.clip(gt_img.astype(np.float32), 0, 1)
    h, w = lq_img.shape[:2]

    mix_type = random.choice(['alpha_blend', 'channel_shuffle', 'patch_mix'])

    if mix_type == 'alpha_blend':
        alpha = random.uniform(0.2, 0.8)
        lq_mix = lq_img[:, ::-1, ...]  # 水平翻转作为混合图像
        gt_mix = gt_img[:, ::-1, ...]
        lq_img = alpha * lq_img + (1 - alpha) * lq_mix
        gt_img = alpha * gt_img + (1 - alpha) * gt_mix

    elif mix_type == 'channel_shuffle':
        # 仅对3通道图像执行通道洗牌
        if lq_img.ndim == 3 and lq_img.shape[2] == 3:
            channels = [0, 1, 2]
            random.shuffle(channels)
            lq_img = lq_img[..., channels]
            gt_img = gt_img[..., channels]
        else:
            pass  # 非3通道图像跳过该操作

    elif mix_type == 'patch_mix':
        patch_h = random.randint(max(1, h // 4), max(1, h // 2))  # 确保patch尺寸至少为1
        patch_w = random.randint(max(1, w // 4), max(1, w // 2))
        top = random.randint(0, h - patch_h)
        left = random.randint(0, w - patch_w)
        # 用翻转后的图像补丁替换
        lq_img[top:top + patch_h, left:left + patch_w, ...] = lq_img[::-1, ::-1, ...][top:top + patch_h,
                                                              left:left + patch_w, ...]
        gt_img[top:top + patch_h, left:left + patch_w, ...] = gt_img[::-1, ::-1, ...][top:top + patch_h,
                                                              left:left + patch_w, ...]

    return np.clip(lq_img, 0, 1), np.clip(gt_img, 0, 1)
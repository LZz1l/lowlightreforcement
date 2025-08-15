import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted
# 导入数据增强函数
from data.augment import random_crop, random_flip, random_rot90


class LOLv2Dataset(Dataset):
    def __init__(self, root_dir, phase='train', real=True, transform=None, resize=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.resize = resize  # 新增：目标尺寸 (h, w)，如(256, 256)
        data_type = 'Real_captured' if real else 'Synthetic'

        # 构建路径并验证存在性
        self.low_dir = os.path.join(root_dir, data_type, phase, 'Low')
        self.gt_dir = os.path.join(root_dir, data_type, phase, 'Normal')
        assert os.path.exists(self.low_dir), f"低光图像目录不存在: {self.low_dir}"
        assert os.path.exists(self.gt_dir), f"正常光图像目录不存在: {self.gt_dir}"

        # 获取并匹配图像文件
        self.low_images = natsorted([f for f in os.listdir(self.low_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.gt_images = natsorted([f for f in os.listdir(self.gt_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.low_images) == len(self.gt_images), \
            f"低光图像数量({len(self.low_images)})与正常光图像数量({len(self.gt_images)})不匹配"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 读取图像并检查有效性
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        low_img = cv2.imread(low_path)
        gt_img = cv2.imread(gt_path)

        # 检查图像读取是否成功
        if low_img is None:
            raise FileNotFoundError(f"无法读取低光图像: {low_path}")
        if gt_img is None:
            raise FileNotFoundError(f"无法读取正常光图像: {gt_path}")

        # 检查图像尺寸匹配
        assert low_img.shape[:2] == gt_img.shape[:2], \
            f"图像尺寸不匹配: {low_path} ({low_img.shape[:2]}) 与 {gt_path} ({gt_img.shape[:2]})"

        # 转为RGB并归一化到[0,1]（保持numpy数组格式用于后续处理）
        low_img = low_img.transpose(2, 0, 1).copy()  # 先转置再复制，确保连续内存
        gt_img = gt_img.transpose(2, 0, 1).copy()
        low_img = torch.from_numpy(low_img)
        gt_img = torch.from_numpy(gt_img)

        # 调整尺寸（如果指定了resize）
        if self.resize is not None:
            h, w = self.resize
            low_img = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_AREA)
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)

        # 训练阶段应用数据增强
        if self.phase == 'train':
            # 随机裁剪（如果需要比resize更小的尺寸增强）
            if self.resize is not None:
                crop_size = min(self.resize[0], self.resize[1]) // 2  # 裁剪尺寸为resize的一半
                if crop_size > 0:
                    low_img, gt_img = random_crop(low_img, gt_img, crop_size)
            # 随机翻转和旋转
            low_img, gt_img = random_flip(low_img, gt_img)
            low_img, gt_img = random_rot90(low_img, gt_img)

        # 检查通道数（确保3通道）
        assert low_img.ndim == 3 and low_img.shape[2] == 3, \
            f"低光图像不是3通道: {low_path} (shape: {low_img.shape})"
        assert gt_img.ndim == 3 and gt_img.shape[2] == 3, \
            f"正常光图像不是3通道: {gt_path} (shape: {gt_img.shape})"

        # 转为Tensor (HWC -> CHW)
        low_img = torch.from_numpy(low_img.transpose(2, 0, 1))
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1))

        sample = {'low': low_img, 'gt': gt_img, 'low_path': low_path, 'gt_path': gt_path}

        if self.transform:
            sample = self.transform(sample)

        return sample
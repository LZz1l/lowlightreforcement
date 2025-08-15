import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset
from data.augment import random_crop, random_flip, random_rot90  # 确保导入正确

class LOLv2Dataset(Dataset):
    def __init__(self, root_dir, phase='train', real=True, transform=None, resize=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.resize = resize  # 目标尺寸 (h, w)，如(256, 256)
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

        # 读取图像（HWC格式，BGR通道）
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        low_img = cv2.imread(low_path)  # shape: (H, W, 3)
        gt_img = cv2.imread(gt_path)    # shape: (H, W, 3)

        # 检查图像读取是否成功
        if low_img is None:
            raise FileNotFoundError(f"无法读取低光图像: {low_path}")
        if gt_img is None:
            raise FileNotFoundError(f"无法读取正常光图像: {gt_path}")

        # 检查原始尺寸匹配
        assert low_img.shape[:2] == gt_img.shape[:2], \
            f"图像尺寸不匹配: {low_path} ({low_img.shape[:2]}) 与 {gt_path} ({gt_img.shape[:2]})"

        # 转为RGB通道（保持HWC格式的numpy数组）
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # 归一化到[0, 1]（numpy数组）
        low_img = low_img.astype(np.float32) / 255.0
        gt_img = gt_img.astype(np.float32) / 255.0

        # 调整尺寸（在numpy阶段处理，确保HWC格式）
        if self.resize is not None:
            h, w = self.resize
            # cv2.resize参数是(w, h)，因为格式是(width, height)
            low_img = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_AREA)
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)

        # 训练阶段数据增强（仍为numpy数组，HWC格式）
        if self.phase == 'train':
            # 随机翻转
            low_img, gt_img = random_flip(low_img, gt_img)
            # 随机旋转90度倍数（旋转后可能改变尺寸，需再次resize确保统一）
            low_img, gt_img = random_rot90(low_img, gt_img)
            # 旋转后强制resize到目标尺寸（关键：确保尺寸统一）
            if self.resize is not None:
                low_img = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_AREA)
                gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)

        # 转为CHW格式并转换为Tensor
        low_img = torch.from_numpy(low_img.transpose(2, 0, 1))  # HWC -> CHW
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1))    # HWC -> CHW

        sample = {'low': low_img, 'gt': gt_img, 'low_path': low_path, 'gt_path': gt_path}

        if self.transform:
            sample = self.transform(sample)

        return sample
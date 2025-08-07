import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted

class LOLv2Dataset(Dataset):
    def __init__(self, root_dir, phase='train', real=True, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录
            phase (string): 'train'或'val'或'test'
            real (bool): 是否使用真实场景数据，False则使用合成数据
            transform (callable, optional): 数据增强函数
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        # 确定数据类型目录
        data_type = 'Real_captured' if real else 'Synthetic'

        # 构建高低光图像路径
        self.low_dir = os.path.join(root_dir, data_type, phase, 'Low')
        self.gt_dir = os.path.join(root_dir, data_type, phase, 'Normal')

        # 获取图像文件名列表
        self.low_images = natsorted([f for f in os.listdir(self.low_dir) if f.endswith(('.png', '.jpg'))])
        self.gt_images = natsorted([f for f in os.listdir(self.gt_dir) if f.endswith(('.png', '.jpg'))])

        # 检查文件数量是否匹配
        assert len(self.low_images) == len(self.gt_images), \
            f"低光图像数量({len(self.low_images)})与正常光图像数量({len(self.gt_images)})不匹配"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 读取图像
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        low_img = cv2.imread(low_path)
        gt_img = cv2.imread(gt_path)

        # 转为RGB并归一化到[0,1]
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0

        # 转为Tensor
        low_img = torch.from_numpy(low_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()

        sample = {'low': low_img, 'gt': gt_img, 'low_path': low_path, 'gt_path': gt_path}

        # 应用数据增强
        if self.transform:
            sample = self.transform(sample)

        return sample
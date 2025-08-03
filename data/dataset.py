import os
import yaml
import numpy as np
import torch.utils.data as data
from basicsr.data.data_util import FileClient, imfrombytes, img2tensor  # 从data_util导入工具
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MultiDataset(data.Dataset):
    """多数据集支持的低光增强数据集（适配BasicSR实际结构）"""

    def __init__(self, opt):
        super(MultiDataset, self).__init__()
        self.opt = opt
        self.dataset_name = opt['dataset_name']  # 数据集名称（如"LOL-v1"）
        self.phase = 'train' if opt['is_train'] else opt['phase']  # train/val/test

        # 加载数据集路径配置（用户填写的dataset_paths.yaml）
        with open(opt['dataset_paths'], 'r', encoding='utf-8') as f:
            self.dataset_paths = yaml.safe_load(f)[self.dataset_name]
        self.root = self.dataset_paths['root']
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"数据集路径不存在：{self.root}，请检查dataset_paths.yaml")

        # 初始化文件读取客户端（BasicSR的FileClient，支持本地/网络文件）
        self.file_client = FileClient(opt['io_backend'], self.root)

        # 获取图像路径列表
        self.low_paths, self.high_paths = self._get_image_paths()
        logger = get_root_logger()
        logger.info(f"加载 {self.dataset_name} {self.phase} 数据集：{len(self.low_paths)} 张图像")

    def _get_image_paths(self):
        """根据数据集名称和阶段生成图像路径"""
        # 拼接低光/正常光图像的相对路径（从dataset_paths.yaml读取）
        if self.phase == 'train':
            low_dir = self.dataset_paths['train_low']
            high_dir = self.dataset_paths['train_high']
        elif self.phase == 'val':
            # 无val集的数据集自动使用test集
            low_dir = self.dataset_paths.get('val_low', self.dataset_paths['test_low'])
            high_dir = self.dataset_paths.get('val_high', self.dataset_paths['test_high'])
        elif self.phase == 'test':
            low_dir = self.dataset_paths['test_low']
            high_dir = self.dataset_paths['test_high']
        else:
            raise ValueError(f"无效阶段：{self.phase}，可选：train/val/test")

        # 生成完整路径并校验
        low_full_dir = os.path.join(self.root, low_dir)
        high_full_dir = os.path.join(self.root, high_dir)
        low_names = sorted(os.listdir(low_full_dir))
        high_names = sorted(os.listdir(high_full_dir))

        if len(low_names) != len(high_names):
            raise ValueError(
                f"{self.dataset_name} {self.phase}集：低光{len(low_names)}张，正常光{len(high_names)}张，数量不匹配")

        return [os.path.join(low_dir, n) for n in low_names], [os.path.join(high_dir, n) for n in high_names]

    def __getitem__(self, index):
        # 读取低光图像和正常光图像（使用BasicSR的data_util工具）
        low_path = self.low_paths[index]
        high_path = self.high_paths[index]

        # 读取图像字节流（FileClient是BasicSR的核心工具，支持多种IO后端）
        low_bytes = self.file_client.get(low_path, 'low_light')
        high_bytes = self.file_client.get(high_path, 'high_light')

        # 转为numpy数组（imfrombytes是BasicSR的工具，处理不同格式图像）
        low_img = imfrombytes(low_bytes, float32=True)  # [0,1] float32
        high_img = imfrombytes(high_bytes, float32=True)

        # 训练阶段数据增强（含图像混合）
        if self.opt['is_train']:
            low_img, high_img = self._augment(low_img, high_img)

        # 转为Tensor（img2tensor是BasicSR的工具，BGR转RGB）
        low_tensor = img2tensor(low_img, bgr2rgb=True, float32=True)
        high_tensor = img2tensor(high_img, bgr2rgb=True, float32=True)

        return {
            'lq': low_tensor, 'gt': high_tensor,
            'lq_path': os.path.join(self.root, low_path),
            'gt_path': os.path.join(self.root, high_path)
        }

    def _augment(self, low_img, high_img):
        """数据增强：裁剪、翻转、旋转、图像混合"""
        from .augment import random_crop, random_flip, random_rot90, image_mixing

        # 随机裁剪
        crop_size = self.opt.get('crop_size', 128)
        low_img, high_img = random_crop(low_img, high_img, crop_size)

        # 随机翻转和旋转
        low_img, high_img = random_flip(low_img, high_img)
        low_img, high_img = random_rot90(low_img, high_img)

        # 图像混合（50%概率）
        if np.random.rand() < 0.5:
            low_img, high_img = image_mixing(low_img, high_img)

        return low_img, high_img

    def __len__(self):
        return len(self.low_paths)
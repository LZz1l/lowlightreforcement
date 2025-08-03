import os
from PIL import Image
import torch.utils.data as data


class LOLv2Dataset(data.Dataset):
    def __init__(self, data_root, phase='train', real=True):
        self.phase = phase
        self.real = real
        self.low_dir = os.path.join(data_root, 'Real_captured' if real else 'Synthetic', phase, 'Low')
        self.gt_dir = os.path.join(data_root, 'Real_captured' if real else 'Synthetic', phase, 'Normal')
        self.file_list = sorted(os.listdir(self.low_dir))

    def __getitem__(self, index):
        low_path = os.path.join(self.low_dir, self.file_list[index])
        gt_path = os.path.join(self.gt_dir, self.file_list[index])

        low = Image.open(low_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        # 简单预处理（可根据需求添加数据增强）
        low = low.resize((256, 256))
        gt = gt.resize((256, 256))

        return {
            'low': torch.from_numpy(np.array(low)).permute(2, 0, 1).float() / 255.,
            'gt': torch.from_numpy(np.array(gt)).permute(2, 0, 1).float() / 255.,
            'filename': self.file_list[index]
        }

    def __len__(self):
        return len(self.file_list)
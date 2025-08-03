import os
import torch
import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import imwrite, scandir
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.logger import get_root_logger
from models.laenet import LAENet
from data.dataset import LAENetDataset
from torch.utils.data import DataLoader


def calculate_dark_region_map(pred, gt, threshold=0.2):
    """计算暗区mAP（自定义指标：低光区域的增强效果）"""
    # 暗区定义：GT中像素值 < threshold 的区域
    dark_mask = (gt < threshold).float()
    if dark_mask.sum() == 0:
        return 1.0  # 无暗区时视为完美

    # 暗区内的PSNR（评估暗区增强效果）
    dark_pred = pred * dark_mask
    dark_gt = gt * dark_mask
    mse = torch.mean((dark_pred - dark_gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 ** 2 / mse)  # 假设像素值范围[0,1]


def test():
    # 配置
    opt = {
        'model_path': './experiments/LAENet_train/models/latest_net_g.pth',
        'dataset_root': './datasets/LOLv2/test',
        'meta_info_file': './datasets/LOLv2/test_meta.txt',
        'save_dir': './results/LAENet'
    }
    logger = get_root_logger()

    # 加载模型
    model = LAENet()
    model.load_state_dict(torch.load(opt['model_path'])['params'])
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # 加载测试集
    dataset = LAENetDataset({
        'root_path': opt['dataset_root'],
        'meta_info_file': opt['meta_info_file'],
        'is_train': False,
        'io_backend': {'type': 'disk'}
    })
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 计算指标
    total_psnr = 0.0
    total_ssim = 0.0
    total_dark_psnr = 0.0
    os.makedirs(opt['save_dir'], exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lq = data['lq'].cuda() if torch.cuda.is_available() else data['lq']
            gt = data['gt'].cuda() if torch.cuda.is_available() else data['gt']
            path = data['lq_path'][0]

            # 推理
            output = model(lq)

            # 计算指标（转为numpy，范围[0,1]）
            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            gt_np = gt.squeeze().cpu().numpy().transpose(1, 2, 0)

            psnr = calculate_psnr(output_np, gt_np, crop_border=0)
            ssim = calculate_ssim(output_np, gt_np, crop_border=0)
            dark_psnr = calculate_dark_region_map(output, gt)

            total_psnr += psnr
            total_ssim += ssim
            total_dark_psnr += dark_psnr

            # 保存结果
            save_path = os.path.join(opt['save_dir'], os.path.basename(path))
            imwrite(output_np * 255, save_path)  # 转为[0,255]保存

            logger.info(
                f'[{i + 1}/{len(dataloader)}] {path} PSNR: {psnr:.2f} SSIM: {ssim:.4f} DarkPSNR: {dark_psnr:.2f}')

    # 平均指标
    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_dark_psnr = total_dark_psnr / len(dataloader)
    logger.info(f'Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}, Average DarkPSNR: {avg_dark_psnr:.2f}')


if __name__ == '__main__':
    test()
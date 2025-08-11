import os
import torch
import numpy as np
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import imwrite
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.logger import get_root_logger
from models.laenet import LAENet
from data.datasets.lolv2_dataset import LOLv2Dataset  # 修正数据集导入路径
from torch.utils.data import DataLoader


def calculate_dark_region_map(pred, gt, threshold=0.2):
    """确保在同一设备上计算"""
    pred = pred.to(gt.device)  # 强制设备一致
    dark_mask = (gt < threshold).float()
    if dark_mask.sum() == 0:
        return 1.0

    dark_pred = pred * dark_mask
    dark_gt = gt * dark_mask
    mse = torch.mean((dark_pred - dark_gt) **2)
    return 10 * torch.log10(1** 2 / mse) if mse != 0 else float('inf')


def test(custom_model=None):
    opt = {
        'model_path': './experiments/LAENet_train/models/latest_net_g.pth',
        'dataset_root': './datasets/LOLv2/test',
        'meta_info_file': './datasets/LOLv2/test_meta.txt',
        'save_dir': './results/LAENet'
    }
    logger = get_root_logger()

    # 加载模型
    model = custom_model if custom_model is not None else LAENet()
    # 加载权重（匹配训练时的保存格式：包含'params'键）
    ckpt = torch.load(opt['model_path'], map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['params'])  # 修正键名
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载测试集（使用正确的数据集类）
    dataset = LOLv2Dataset({
        'dataroot': opt['dataset_root'],
        'phase': 'test',
        'real': True,
        'io_backend': {'type': 'disk'}
    })
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_psnr = 0.0
    total_ssim = 0.0
    total_dark_psnr = 0.0
    os.makedirs(opt['save_dir'], exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lq = data['low'].to(device)
            gt = data['gt'].to(device)  # 移至与模型相同设备
            path = data['lq_path'][0]

            output = model(lq)

            # 转为numpy（修正通道顺序：[C,H,W]→[H,W,C]）
            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            gt_np = gt.squeeze().cpu().numpy().transpose(1, 2, 0)

            psnr = calculate_psnr(output_np, gt_np, crop_border=0)
            ssim = calculate_ssim(output_np, gt_np, crop_border=0)
            dark_psnr = calculate_dark_region_map(output, gt)  # 传入tensor计算

            total_psnr += psnr
            total_ssim += ssim
            total_dark_psnr += dark_psnr

            # 保存结果（确保范围正确）
            save_path = os.path.join(opt['save_dir'], os.path.basename(path))
            imwrite((output_np * 255).astype(np.uint8), save_path)

            logger.info(
                f'[{i + 1}/{len(dataloader)}] {path} '
                f'PSNR: {psnr:.2f} SSIM: {ssim:.4f} DarkPSNR: {dark_psnr:.2f}')

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_dark_psnr = total_dark_psnr / len(dataloader)
    logger.info(f'Average PSNR: {avg_psnr:.2f} dB, '
                f'Average SSIM: {avg_ssim:.4f}, '
                f'Average DarkPSNR: {avg_dark_psnr:.2f}')


if __name__ == '__main__':
    test()
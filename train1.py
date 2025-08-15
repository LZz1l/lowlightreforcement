import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from basicsr.metrics import calculate_psnr, calculate_ssim  # 新增指标计算工具
import numpy as np
from data.datasets.lolv2_dataset import LOLv2Dataset
from models.laenet import LAENet
from train.losses import RetinexLoss

# 配置参数
config = {
    'data_root': 'C:/Users/ASUS/OneDrive/Desktop/LOLv2',
    'batch_size': 2,  # 从8降至2（根据GPU显存调整）
    'lr': 1e-4,
    'epochs': 50,
    'save_dir': './checkpoints',
    'image_size': (128, 128)  # 从256x256降至128x128（或192x192）
}

# 创建保存目录
os.makedirs(config['save_dir'], exist_ok=True)

# 初始化设备、模型、优化器和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LAENet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
# 新增学习率调度器（余弦退火）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config['epochs'], eta_min=1e-6
)
criterion = RetinexLoss(alpha=0.5, beta=0.5).to(device)

# 数据加载（添加resize参数统一图像尺寸）
train_dataset = LOLv2Dataset(
    config['data_root'],
    phase='train',
    real=True,
    resize=config['image_size']  # 传入统一尺寸
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = LOLv2Dataset(
    config['data_root'],
    phase='Test',
    real=True,
    resize=config['image_size']  # 传入统一尺寸
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# 训练循环
best_val_loss = float('inf')  # 初始化最佳验证损失
for epoch in range(config['epochs']):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        low = batch['low'].to(device)
        gt = batch['gt'].to(device)

        # 前向传播（同时计算L和R，避免重复编码）
        output = model(low)
        L = model.L  # 直接从模型获取光照分量（需LAENet支持）
        R = model.R  # 直接从模型获取反射分量（需LAENet支持）

        # 计算损失并反向传播
        loss = criterion(L, R, low,gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * low.size(0)

    # 计算平均训练损失并更新学习率
    avg_train_loss = train_loss / len(train_dataset)
    scheduler.step()  # 每个epoch更新学习率

    # 验证过程
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low = batch['low'].to(device)
            gt = batch['gt'].to(device)
            output = model(low)

            # 计算验证指标
            l1_loss = nn.L1Loss()(output, gt).item()
            # 转换为numpy格式计算PSNR/SSIM（需匹配数据范围）
            output_np = output.detach().cpu().numpy()  # 形状为[B, C, H, W]
            output_np = np.transpose(output_np, (0, 2, 3, 1))  # 转为[B, H, W, C]
            output_np = output_np.squeeze(0)  # 移除批次维度（B=1），变为[H, W, C]
            output_np = (output_np * 255).astype(np.uint8)

            gt_np = gt.detach().cpu().numpy()  # 形状为[B, C, H, W]
            gt_np = np.transpose(gt_np, (0, 2, 3, 1))  # 转为[B, H, W, C]
            gt_np = gt_np.squeeze(0)  # 移除批次维度，变为[H, W, C]
            gt_np = (gt_np * 255).astype(np.uint8)
            psnr = calculate_psnr(output_np, gt_np, crop_border=0)
            ssim = calculate_ssim(output_np, gt_np, crop_border=0)

            val_loss += l1_loss * low.size(0)
            val_psnr += psnr * low.size(0)
            val_ssim += ssim * low.size(0)

    # 计算平均验证指标
    avg_val_loss = val_loss / len(val_dataset)
    avg_val_psnr = val_psnr / len(val_dataset)
    avg_val_ssim = val_ssim / len(val_dataset)

    # 打印训练日志
    print(
        f'Epoch {epoch + 1}/{config["epochs"]}\n'
        f'Train Loss: {avg_train_loss:.4f}\n'
        f'Val Loss: {avg_val_loss:.4f} | '
        f'Val PSNR: {avg_val_psnr:.2f} dB | '
        f'Val SSIM: {avg_val_ssim:.4f}\n'
        f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
    )

    # 保存最佳模型（仅保存验证损失最低的模型）
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = f'{config["save_dir"]}/best_model.pth'
        torch.save({
            'params': model.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss
        }, save_path)
        print(f'Saved best model to {save_path}\n')

# 训练结束后保存最终模型
final_save_path = f'{config["save_dir"]}/final_model.pth'
torch.save({'params': model.state_dict(), 'epoch': config['epochs']}, final_save_path)
print(f'Training completed. Final model saved to {final_save_path}')


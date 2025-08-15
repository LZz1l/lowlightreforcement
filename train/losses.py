import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(RetinexLoss, self).__init__()
        self.alpha = alpha  # 光照平滑损失权重
        self.beta = beta  # 反射保真损失权重
        self.l1_loss = nn.L1Loss()

        # 定义Sobel算子（用于梯度计算，保持尺寸）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)  # x方向（水平梯度）
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)  # y方向（垂直梯度）

    def forward(self, L, R, low, gt):
        # 1. 检查所有输入尺寸是否一致（调试用）
        assert L.shape == R.shape == low.shape == gt.shape, \
            f"尺寸不匹配: L={L.shape}, R={R.shape}, low={low.shape}, gt={gt.shape}"

        # 2. 物理一致性损失：low ≈ L ⊙ R（像素-wise乘积）
        recon_loss = self.l1_loss(L * R, low)

        # 3. 光照分量平滑损失（使用Sobel卷积计算梯度，保持尺寸）
        # 扩展Sobel算子以匹配输入通道数
        B, C = L.shape[0], L.shape[1]
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1).to(L.device)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1).to(L.device)

        # 计算梯度（使用padding='same'保持尺寸）
        l_grad_x = F.conv2d(L, sobel_x, padding=1, groups=C)  # 水平梯度
        l_grad_y = F.conv2d(L, sobel_y, padding=1, groups=C)  # 垂直梯度
        smooth_loss = torch.mean(torch.abs(l_grad_x) + torch.abs(l_grad_y))

        # 4. 反射分量保真损失：R应接近gt/L（避免除零）
        safe_L = torch.clamp(L, min=0.01)  # 防止除以零
        reflect_loss = self.l1_loss(R, gt / safe_L)

        # 总损失
        total_loss = recon_loss + self.alpha * smooth_loss + self.beta * reflect_loss
        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexPerturbationLoss(nn.Module):
    """Retinex理论损失函数（带尺寸检查）"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1 = nn.L1Loss()

    def forward(self, L, R, x):
        # 严格检查尺寸匹配（提前发现问题）
        assert L.shape == x.shape, f"L尺寸 {L.shape} 与输入x尺寸 {x.shape} 不匹配"
        assert R.shape == x.shape, f"R尺寸 {R.shape} 与输入x尺寸 {x.shape} 不匹配"

        # 1. 重构损失：L*R ≈ 输入低光图像x
        recon_loss = self.l1(L * R, x)

        # 2. 光照平滑损失：光照分量空间变化缓慢
        l_grad = (torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :])  # 垂直方向梯度
                  + torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1]))  # 水平方向梯度
        smooth_loss = l_grad.mean()

        # 3. 反射分量非负损失：物理上反射率≥0
        neg_r_loss = torch.clamp(-R, min=0).mean()

        # 总损失
        total_loss = recon_loss + 0.1 * smooth_loss + 0.05 * neg_r_loss
        return self.loss_weight * total_loss
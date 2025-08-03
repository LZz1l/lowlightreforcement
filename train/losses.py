import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.losses import L1Loss, PerceptualLoss  # 复用BasicSR基础损失


@LOSS_REGISTRY.register()
class RetinexPerturbationLoss(nn.Module):
    """Retinex扰动损失：约束光照L和反射R的物理合理性"""

    def __init__(self, loss_weight=1.0):
        super(RetinexPerturbationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.l1 = L1Loss()

    def forward(self, L, R, x):
        """
        Args:
            L: 光照分量 [B, 3, H, W]
            R: 反射分量 [B, 3, H, W]
            x: 输入低光图像 [B, 3, H, W]
        """
        # 1. 约束 L * R ≈ x（低光图像生成公式）
        recon_loss = self.l1(L * R, x)

        # 2. 约束光照L平滑（低光图像光照变化缓慢）
        l_grad = torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :]) + torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1])
        smooth_loss = l_grad.mean()

        # 3. 约束反射R非负（物理意义：反射率≥0）
        neg_r_loss = torch.clamp(-R, min=0).mean()

        return self.loss_weight * (recon_loss + 0.1 * smooth_loss + 0.05 * neg_r_loss)


@LOSS_REGISTRY.register()
class DistillationLoss(nn.Module):
    """蒸馏损失：从教师模型（如Retinexformer）蒸馏知识"""

    def __init__(self, loss_weight=1.0, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature  # 蒸馏温度，控制软化程度
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: 学生模型特征 [B, C, H, W]
            teacher_feat: 教师模型特征 [B, C, H, W]
        """
        # 特征蒸馏（MSE损失）
        feat_loss = self.mse(student_feat, teacher_feat)

        # 若有输出概率分布，可添加软化后的KL散度损失（根据模型结构扩展）
        return self.loss_weight * feat_loss
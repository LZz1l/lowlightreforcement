import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import ARCH_REGISTRY
from models.modules.iga_block import IGABlock


@ARCH_REGISTRY.register()
class LAENet(nn.Module):
    """低光增强网络LAENet（修复尺寸不匹配问题）"""

    def __init__(self, base_channels=32):
        super().__init__()
        # 编码器（Retinex分解）
        self.retinex_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            IGABlock(base_channels),  # 注意力模块
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            IGABlock(base_channels * 2)
        )

        # 光照分量分解
        self.decompose_L = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 显式指定align_corners
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 光照分量在[0,1]
        )

        # 反射分量分解
        self.decompose_R = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 反射分量在[0,1]
        )

        # 最终增强输出
        self.enhance_head = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=3, padding=1),  # 融合L和R（3+3=6通道）
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x: 低光图像 [B,3,H,W]
        B, C, H, W = x.shape  # 记录原始尺寸
        retinex_feat = self.retinex_encoder(x)  # 特征提取

        # 分解分量并强制上采样至原始尺寸（核心修复）
        L = self.decompose_L(retinex_feat)
        R = self.decompose_R(retinex_feat)
        self.L = F.interpolate(L, size=(H, W), mode='bilinear', align_corners=True)  # 确保与输入同尺寸
        self.R = F.interpolate(R, size=(H, W), mode='bilinear', align_corners=True)

        # 物理约束：光照分量避免过暗（防止反射分量异常）
        self.L = torch.clamp(self.L, min=0.01, max=1.0)

        # 融合输出
        output = self.enhance_head(torch.cat([self.L, self.R], dim=1))
        return output
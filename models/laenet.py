import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY
from models.modules.iga_block import IGABlock


@ARCH_REGISTRY.register()
class LAENet(nn.Module):
    """低光增强网络LAENet"""

    def __init__(self, base_channels=32):
        super().__init__()
        # 编码器（Retinex分解）
        self.retinex_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            IGABlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            IGABlock(base_channels * 2)
        )

        # 光照分量分解
        self.decompose_L = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 光照分量在[0,1]
        )

        # 反射分量分解
        self.decompose_R = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 反射分量在[0,1]
        )

        # 最终增强输出
        self.enhance_head = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=3, padding=1),  # 融合L和R
            nn.ReLU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x: 低光图像 [B,3,H,W]
        retinex_feat = self.retinex_encoder(x)  # 特征提取
        self.L = self.decompose_L(retinex_feat)  # 保存为实例属性self.L
        self.R = self.decompose_R(retinex_feat)  # 保存为实例属性self.R
        output = self.enhance_head(torch.cat([self.L, self.R], dim=1))  # 融合输出
        return output
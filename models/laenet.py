import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY  # 导入自定义注册器
from models.modules.iga_block import IGABlock  # 导入优化后的注意力模块


@ARCH_REGISTRY.register()  # 使用自定义注册器
class LAENet(nn.Module):
    """低光增强网络LAENet"""

    def __init__(self, base_channels=32):
        super().__init__()
        # 编码器（Retinex分解）
        self.retinex_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            IGABlock(base_channels),  # 使用内存优化的注意力模块
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
        L = self.decompose_L(retinex_feat)  # 光照分量
        R = self.decompose_R(retinex_feat)  # 反射分量
        output = self.enhance_head(torch.cat([L, R], dim=1))  # 融合输出
        return output

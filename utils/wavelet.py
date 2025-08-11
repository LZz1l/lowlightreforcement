import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class WaveletTransform(nn.Module):
    """小波变换模块（用于光照分量处理，适配LAENet）"""
    def __init__(self, level: int = 3, wavelet_type: str = 'db1'):
        super().__init__()
        self.level = level
        self.wavelet_type = wavelet_type
        self.kernel = self._get_wavelet_kernel()  # 小波卷积核（水平、垂直、对角线）

    def _get_wavelet_kernel(self) -> torch.Tensor:
        """获取小波基卷积核（以db1为例）"""
        if self.wavelet_type == 'db1':
            # db1小波的低通和高通滤波器
            low = np.array([1, 1]) / np.sqrt(2)
            high = np.array([1, -1]) / np.sqrt(2)
        else:
            raise NotImplementedError(f"暂不支持小波类型: {self.wavelet_type}")

        # 构建2D卷积核（水平、垂直、对角线）
        kernel = []
        kernel.append(np.outer(low, low))  # 低频
        kernel.append(np.outer(low, high))  # 水平高频
        kernel.append(np.outer(high, low))  # 垂直高频
        kernel.append(np.outer(high, high))  # 对角线高频

        # 转为PyTorch卷积核 [out_ch, in_ch, kH, kW]
        kernel = np.stack(kernel)[:, np.newaxis, ...]  # 扩展输入通道维度
        return torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        正向小波变换（多尺度分解）

        Args:
            x: 输入特征图 [B, C, H, W]（如光照分量L）

        Returns:
            低频分量和各尺度高频分量列表
        """
        B, C, H, W = x.shape
        kernel = self.kernel.to(x.device)  # 移动到输入设备（GPU/CPU）
        outputs = []

        current = x
        for _ in range(self.level):
            # 对每个通道应用小波变换
            chs = []
            for c in range(C):
                # 卷积实现小波变换（步长2下采样）
                feat = nn.functional.conv2d(
                    current[:, c:c+1, ...],
                    kernel,
                    stride=2,
                    padding=0
                )
                chs.append(feat)
            # 合并通道 [B, C*4, H/2, W/2]
            current = torch.cat(chs, dim=1)
            # 分离低频（前C通道）和高频（后3C通道）
            low = current[:, :C, ...]
            high = current[:, C:, ...]
            outputs.append(high)
            current = low

        outputs.append(low)  # 最后一层低频分量
        return low, outputs

    def inverse(self, low: torch.Tensor, highs: list) -> torch.Tensor:
        """
        小波逆变换（重构）

        Args:
            low: 最后一层低频分量 [B, C, H, W]
            highs: 各尺度高频分量列表

        Returns:
            重构后的特征图 [B, C, H', W']
        """
        current = low
        kernel = self.kernel.to(current.device)
        C = current.shape[1]

        for high in reversed(highs):
            # 合并低频和高频 [B, C*4, H, W]
            current = torch.cat([current, high], dim=1)
            # 对每个通道应用逆变换（步长1，填充1，上采样2倍）
            chs = []
            for c in range(C):
                feat = nn.functional.conv_transpose2d(
                    current[:, c::C, ...],  # 提取第c个通道的4个分量
                    kernel,
                    stride=2,
                    padding=0,
                    output_padding=1  # 调整输出尺寸
                )
                chs.append(feat)
            current = torch.sum(torch.cat(chs, dim=1), dim=1, keepdim=True)  # 合并通道

        return current


# 适配LAENet中的wavelet_processor
def create_wavelet_processor(level: int = 3) -> WaveletTransform:
    """创建小波处理器（用于光照分量优化）"""
    return WaveletTransform(level=level)

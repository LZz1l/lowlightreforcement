import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增导入
import pywt
from basicsr.utils.registry import ARCH_REGISTRY
from .modules.iga_block import IGABlock
from .modules.hfrm_pro import HFRMPro


@ARCH_REGISTRY.register()
class LAENet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3,
                 base_channels=64, wavelet_level=3):
        super(LAENet, self).__init__()
        self.wavelet_level = wavelet_level

        # 1. Retinex分支：分解光照L和反射R
        self.retinex_encoder = nn.Sequential(
            nn.Conv2d(num_in_ch, base_channels, kernel_size=3, padding=1),
            IGABlock(base_channels),  # 全局注意力块
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        )
        # 输出光照分量L和反射分量R
        self.decompose_L = nn.Conv2d(base_channels * 2, num_out_ch, kernel_size=3, padding=1)
        self.decompose_R = nn.Conv2d(base_channels * 2, num_out_ch, kernel_size=3, padding=1)

        # 2. 小波扩散分支：处理光照分量L
        self.wavelet_processor = nn.Sequential(
            HFRMPro(num_out_ch, base_channels),  # 动态空洞残差块
            HFRMPro(base_channels, base_channels),
            nn.Conv2d(base_channels, num_out_ch, kernel_size=3, padding=1)
        )

        # 3. 融合模块：整合R和优化后的L
        self.fusion = nn.Conv2d(num_out_ch * 2, num_out_ch, kernel_size=3, padding=1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: 低光图像 [B, 3, H, W]，假设已归一化到[0,1]

        # 步骤1：Retinex分解
        retinex_feat = self.retinex_encoder(x)
        L = self.decompose_L(retinex_feat)  # 光照分量
        R = self.decompose_R(retinex_feat)  # 反射分量（保留细节）

        # 确保光照分量在合理范围内
        L = torch.clamp(L, 0.01, 1.0)  # 避免光照过暗导致数值不稳定

        # 步骤2：小波分解+扩散（仅处理光照分量L）
        # 使用PyTorch实现小波变换，避免CPU/GPU数据传输
        L_processed = self.wavelet_processor(L)

        # 步骤3：融合反射分量R和优化后的光照L
        output = self.fusion(torch.cat([R, L_processed], dim=1))

        # 确保输出在[0,1]范围内
        return torch.clamp(output, 0.0, 1.0)
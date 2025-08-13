import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增导入
from utils.registry import ARCH_REGISTRY
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

        # 2. 小波扩散分支：处理光照分量L（移除CPU小波变换，改用纯GPU操作）
        self.wavelet_processor = nn.Sequential(
            HFRMPro(num_out_ch, base_channels),  # 动态空洞残差块
            HFRMPro(base_channels, base_channels),
            nn.Conv2d(base_channels, num_out_ch, kernel_size=3, padding=1)
        )

        # 3. 融合模块：整合R和优化后的L
        self.fusion = nn.Conv2d(num_out_ch * 2, num_out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # x: 低光图像 [B, 3, H, W]（GPU上的tensor）

        # 步骤1：Retinex分解
        retinex_feat = self.retinex_encoder(x)
        L = self.decompose_L(retinex_feat)  # 光照分量
        R = self.decompose_R(retinex_feat)  # 反射分量（保留细节）

        # 约束光照分量在合理范围（避免数值不稳定）
        L = torch.clamp(L, 0.01, 1.0)  # 光照不能过暗（>0.01）或过亮（<1.0）
        R = torch.clamp(R, 0.0, 1.0)   # 反射率非负

        # 步骤2：处理光照分量L（移除CPU小波变换，直接用网络处理）
        L_processed = self.wavelet_processor(L)
        L_processed = torch.clamp(L_processed, 0.01, 1.0)  # 保持光照合理性

        # 步骤3：融合反射分量R和优化后的光照L
        output = self.fusion(torch.cat([R, L_processed], dim=1))
        return torch.clamp(output, 0.0, 1.0)  # 确保输出在[0,1]范围内


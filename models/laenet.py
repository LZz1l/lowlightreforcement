import torch
import torch.nn as nn
import pywt
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

        # 2. 小波扩散分支：处理光照分量L
        self.wavelet_processor = nn.Sequential(
            HFRMPro(num_out_ch, base_channels),  # 动态空洞残差块
            HFRMPro(base_channels, base_channels),
            nn.Conv2d(base_channels, num_out_ch, kernel_size=3, padding=1)
        )

        # 3. 融合模块：整合R和优化后的L
        self.fusion = nn.Conv2d(num_out_ch * 2, num_out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # x: 低光图像 [B, 3, H, W]

        # 步骤1：Retinex分解
        retinex_feat = self.retinex_encoder(x)
        L = self.decompose_L(retinex_feat)  # 光照分量
        R = self.decompose_R(retinex_feat)  # 反射分量（保留细节）

        # 步骤2：小波分解+扩散（仅处理光照分量L）
        # 小波分解（PyWavelets，CPU操作，需转为numpy）
        L_np = L.detach().cpu().numpy()  # [B, 3, H, W]
        wavelet_coeffs = []
        for b in range(L_np.shape[0]):
            for c in range(3):
                coeffs = pywt.wavedec2(L_np[b, c], 'db4', level=self.wavelet_level)
                wavelet_coeffs.append(coeffs)
        # 仅对低频分量进行扩散处理（简化：这里用网络直接处理L的小波低频）
        L_processed = self.wavelet_processor(L)  # 模拟小波域优化

        # 步骤3：融合反射分量R和优化后的光照L
        output = self.fusion(torch.cat([R, L_processed], dim=1))
        return output  # 增强后的图像
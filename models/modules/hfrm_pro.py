import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增导入


class HFRMPro(nn.Module):
    """HFRM-Pro: 动态空洞残差块"""

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5]):
        super(HFRMPro, self).__init__()
        self.dilation_rates = dilation_rates
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilation_rates),
                      kernel_size=3, padding=d, dilation=d)
            for d in dilation_rates
        ])
        self.gate = nn.Conv2d(in_channels, len(dilation_rates), kernel_size=1, padding=0)  # 动态选择权重
        self.relu = nn.ReLU(inplace=True)
        self.conv_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.residual = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 动态空洞率选择（基于输入特征的门控机制）
        gate_weights = F.softmax(self.gate(x), dim=1)  # [B, num_rates, H, W]

        # 多空洞卷积分支
        outputs = []
        for i, conv in enumerate(self.convs):
            branch = conv(x)
            outputs.append(branch * gate_weights[:, i:i + 1, ...])  # 按权重加权

        # 融合分支
        fused = torch.cat(outputs, dim=1)
        fused = self.conv_fuse(fused)
        fused = self.relu(fused)

        # 残差连接
        return fused + self.residual(x)
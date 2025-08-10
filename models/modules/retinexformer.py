import torch
import torch.nn as nn
from einops import rearrange


class Retinexformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        # TransformerEncoderLayer参数校验：d_model=64需能被nhead=4整除（64/4=16）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True),  # 显式指定batch_first=True
            num_layers=4
        )
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        # x: [B, 3, H, W]
        b, c, h, w = x.shape
        assert h > 0 and w > 0, f"输入图像尺寸({h}x{w})无效"

        x = self.conv_in(x)  # [B, 64, H, W]
        # 转为序列格式 [B, N, C]，其中N=H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        # 校验序列长度是否为正
        n = x.shape[1]
        assert n > 0, f"序列长度为0，输入尺寸过小(h={h}, w={w})"

        x = self.transformer(x)  # [B, N, 64]
        # 恢复为特征图格式 [B, 64, H, W]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.conv_out(x)  # [B, 3, H, W]
import torch
import torch.nn as nn
from einops import rearrange


class Retinexformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=256),
            num_layers=4
        )
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')  # 转换为序列
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.conv_out(x)
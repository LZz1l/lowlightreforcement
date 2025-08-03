import torch
import torch.nn as nn
import torch.nn.functional as F


class IGABlock(nn.Module):
    """IGA Block: 整合IG-MSA和通道混洗"""

    def __init__(self, dim, num_heads=4):
        super(IGABlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 1. 全局IG-MSA注意力（简化版）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)  # 生成QKV
        self.attn_softmax = nn.Softmax(dim=-1)

        # 2. 局部通道混洗
        self.channel_shuffle = nn.Conv2d(dim, dim, kernel_size=1, groups=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, dim, H, W]
        B, C, H, W = x.shape

        # IG-MSA注意力
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.attn_softmax(attn)
        attn_out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)

        # 残差+通道混洗
        x = x + attn_out
        x = self.channel_shuffle(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # FFN
        x = x + self.ffn(x)
        return x
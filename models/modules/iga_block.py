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
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        # 1. 全局IG-MSA注意力
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)  # 生成QKV
        self.attn_softmax = nn.Softmax(dim=-1)

        # 2. 局部通道混洗
        self.channel_shuffle = nn.Conv2d(dim, dim, kernel_size=1, groups=num_heads)
        self.norm = nn.LayerNorm(dim)  # 作用于通道维度（最后一维）
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, dim, H, W]
        B, C, H, W = x.shape
        assert C == self.dim, f"输入通道数{C}与模块维度{self.dim}不匹配"

        # IG-MSA注意力
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]，N=H*W
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # 缩放点积注意力
        attn = self.attn_softmax(attn)
        attn_out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)  # 重组为[B, C, H, W]

        # 残差+通道混洗
        x = x + attn_out  # 残差连接
        x = self.channel_shuffle(x)  # 通道混洗
        # LayerNorm：需将通道维度放到最后 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # 恢复为[B, C, H, W]

        # FFN
        x = x + self.ffn(x)  # 残差连接
        return x
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
        B, C, H, W = x.shape
        # 添加下采样减少空间维度（例如缩小为1/2）
        x_down = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        B, C, H_down, W_down = x_down.shape  # 新的尺寸

        # 后续注意力计算基于下采样后的特征图
        qkv = self.qkv(x_down).reshape(B, H_down * W_down, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        q, k, v = qkv.unbind(0)  # 每个形状为 [B, num_heads, H_down*W_down, head_dim]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # 此时H*W变为原来的1/4，内存需求大幅降低
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, H_down, W_down, C)
        x_attn = x_attn.permute(0, 3, 1, 2)  # [B, C, H_down, W_down]
        # 上采样回原始尺寸
        x_attn = F.interpolate(x_attn, size=(H, W), mode='bilinear', align_corners=False)

        x = x + self.proj_drop(self.proj(x_attn))
        return x
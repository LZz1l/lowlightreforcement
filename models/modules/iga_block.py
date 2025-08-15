import torch
import torch.nn as nn
import torch.nn.functional as F


class IGABlock(nn.Module):
    """IGA Block: 整合IG-MSA和通道混洗（修复功能版）"""

    def __init__(self, dim, num_heads=4, attn_drop=0.1, proj_drop=0.1):
        super(IGABlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        # 1. 全局IG-MSA注意力（恢复核心组件）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)  # 生成QKV
        self.attn_drop = nn.Dropout(attn_drop)  # 缺失的注意力dropout
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)  # 缺失的投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影后dropout

        # 2. 局部通道混洗（恢复并启用）
        self.channel_shuffle = nn.Conv2d(dim, dim, kernel_size=1, groups=num_heads)
        self.norm = nn.LayerNorm(dim)  # 作用于通道维度

        # 3. FFN前馈网络（恢复并启用）
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape  # 原始输入形状 [B, C, H, W]
        shortcut = x  # 残差连接 shortcut

        # 步骤1: 下采样减少空间维度（保留内存优化）
        x_down = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        B, C, H_down, W_down = x_down.shape  # 下采样后形状 [B, C, H/2, W/2]

        # 步骤2: 计算IG-MSA注意力
        # 生成QKV并调整维度 [B, 3*C, H_down, W_down] -> 3*[B, num_heads, H_down*W_down, head_dim]
        qkv = self.qkv(x_down).reshape(B, H_down * W_down, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # q, k, v形状均为 [B, num_heads, N, head_dim]，其中N=H_down*W_down

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim **-0.5)  # 注意力分数 [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用注意力dropout

        # 注意力加权求和并重塑
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H_down, W_down, C)  # [B, H_down, W_down, C]
        x_attn = x_attn.permute(0, 3, 1, 2)  # 转回 [B, C, H_down, W_down]

        # 投影+dropout
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # 上采样回原始尺寸
        x_attn = F.interpolate(x_attn, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]

        # 步骤3: 残差连接 + 通道混洗
        x = shortcut + x_attn  # 注意力残差
        x = self.channel_shuffle(x)  # 启用通道混洗（修复功能点）

        # 步骤4: 归一化 + FFN
        # 调整维度以适配LayerNorm（[B, C, H, W] -> [B, H, W, C]）
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)  # 通道维度归一化
        x = x.permute(0, 3, 1, 2)  # 转回 [B, C, H, W]

        # FFN前馈网络 + 残差
        x = x + self.ffn(x)  # 完整FFN残差连接（修复功能点）

        return x
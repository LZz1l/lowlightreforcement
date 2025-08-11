import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional


def visualize_attention(
    attention_map: torch.Tensor,
    img_path: str,
    save_path: str,
    alpha: float = 0.5,
    normalize: bool = True
) -> None:
    """
    可视化注意力热力图（适配IGA Block的注意力输出）

    Args:
        attention_map: [H, W] 或 [1, H, W] 注意力权重张量（支持PyTorch张量）
        img_path: 原始低光图像路径
        save_path: 可视化结果保存路径
        alpha: 热力图与原图的融合系数（0~1）
        normalize: 是否对注意力图进行归一化（默认True）
    """
    # 处理注意力图格式（转为numpy数组）
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    # 移除单通道维度（如[1, H, W] -> [H, W]）
    if attention_map.ndim == 3 and attention_map.shape[0] == 1:
        attention_map = attention_map.squeeze(0)
    if attention_map.ndim != 2:
        raise ValueError(f"注意力图维度需为2D，实际为{attention_map.ndim}D")

    # 读取并预处理原图
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB格式（与PyTorch一致）

    # 注意力图归一化和尺寸匹配
    if normalize:
        attn_min, attn_max = attention_map.min(), attention_map.max()
        if attn_max - attn_min < 1e-8:
            attention_map = np.zeros_like(attention_map)
        else:
            attention_map = (attention_map - attn_min) / (attn_max - attn_min)
    attn_resized = cv2.resize(attention_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 生成热力图并融合
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]  # 取RGB通道
    heatmap = (heatmap * 255).astype(np.uint8)
    fused = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # 保存可视化结果
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img), plt.title('Original Image'), plt.axis('off')
    plt.subplot(132), plt.imshow(heatmap), plt.title('Attention Heatmap'), plt.axis('off')
    plt.subplot(133), plt.imshow(fused), plt.title('Fused Result'), plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'注意力热力图已保存至: {save_path}')
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_attention(attention_map, img_path, save_path, alpha=0.5):
    """
    可视化注意力热力图
    attention_map: [H, W] 注意力权重
    img_path: 原始图像路径
    save_path: 保存路径
    alpha: 热力图与原图的融合系数
    """
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 注意力图归一化并resize到原图尺寸
    attn_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    attn_resized = cv2.resize(attn_norm, (img.shape[1], img.shape[0]))

    # 生成热力图
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]  # 转为RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # 与原图融合
    fused = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # 保存
    plt.figure(figsize=(10, 10))
    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.subplot(132), plt.imshow(heatmap), plt.title('Attention')
    plt.subplot(133), plt.imshow(fused), plt.title('Fused')
    plt.savefig(save_path)
    plt.close()
    print(f'注意力热力图已保存至: {save_path}')
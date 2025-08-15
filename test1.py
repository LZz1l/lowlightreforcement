import os
import cv2
import numpy as np
import torch
from models.laenet import LAENet  # 导入修复后的模型


def enhance_image(low_path, model_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 检查文件存在性
    if not os.path.exists(low_path):
        raise FileNotFoundError(f"低光图像文件不存在: {low_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LAENet().eval().to(device)

    # 加载模型（匹配训练时的保存格式）
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['params'])  # 假设训练时保存的键为'params'
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

    # 读取并预处理图像
    low = cv2.imread(low_path)
    if low is None:
        raise ValueError(f"无法读取图像: {low_path}")
    h, w = low.shape[:2]  # 保留原始尺寸
    low_rgb = low[:, :, ::-1]  # BGR转RGB
    low_tensor = torch.from_numpy(low_rgb).permute(2, 0, 1).float() / 255.0  # 归一化并转张量
    low_tensor = low_tensor.unsqueeze(0).to(device)  # 增加批次维度

    # 推理
    with torch.no_grad():
        enhanced = model(low_tensor).squeeze().cpu().numpy()  # 移除批次维度

    # 后处理并保存
    enhanced = enhanced.transpose(1, 2, 0)  # [C,H,W]→[H,W,C]
    enhanced = cv2.resize(enhanced, (w, h))  # 确保与原始尺寸一致
    enhanced = (enhanced * 255).astype(np.uint8)[:, :, ::-1]  # RGB转BGR并转uint8
    cv2.imwrite(save_path, enhanced)
    print(f"增强图像已保存至: {save_path}")


# 测试示例
if __name__ == "__main__":
    enhance_image(
        low_path="test_images/low_light.jpg",  # 输入低光图像路径
        model_path="checkpoints/laenet_best.pth",  # 模型权重路径
        save_path="results/enhanced.jpg"  # 增强图像保存路径
    )
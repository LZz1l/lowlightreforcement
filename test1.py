import os  # 开头导入os
import cv2
import numpy as np
import torch

from models.laenet import LAENet  # 改为测试LAENet


def enhance_image(low_path, model_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(low_path):
        raise FileNotFoundError(f"低光图像文件不存在: {low_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LAENet().eval().to(device)

    # 修正模型加载（匹配训练时的保存格式）
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['params'])  # 使用'params'键
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

    # 读取图像（保留原始尺寸，不强制 resize）
    low = cv2.imread(low_path)
    if low is None:
        raise ValueError(f"无法读取图像: {low_path}")

    h, w = low.shape[:2]  # 记录原始尺寸
    low = low[:, :, ::-1]  # BGR→RGB
    low_tensor = torch.from_numpy(low).permute(2, 0, 1).float() / 255.0
    low_tensor = low_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(low_tensor).squeeze().cpu().numpy()

    # 转回原始尺寸并保存
    enhanced = enhanced.transpose(1, 2, 0)  # [C,H,W]→[H,W,C]
    enhanced = cv2.resize(enhanced, (w, h))  # 恢复原始尺寸
    enhanced = (enhanced * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR
    cv2.imwrite(save_path, enhanced)
    print(f"增强图像已保存至: {save_path}")


if __name__ == "__main__":
    enhance_image(
        'C:/Users/ASUS/OneDrive/Desktop/LOLv2/Real_captured/Test/Low/00690.png',
        'checkpoints/epoch_50.pth',
        'results/enhanced_00690.png'
    )
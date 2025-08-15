import os
import cv2
import numpy as np
import torch
from models.laenet import LAENet  # 确保LAENet类正确定义


def enhance_image(low_path, model_path, save_path):
    # 创建保存目录（确保父目录存在）
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 检查文件存在性（显示绝对路径便于排查）
    if not os.path.exists(low_path):
        raise FileNotFoundError(
            f"低光图像文件不存在！\n绝对路径: {os.path.abspath(low_path)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在！\n绝对路径: {os.path.abspath(model_path)}")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型并加载权重
    model = LAENet().eval().to(device)
    try:
        ckpt = torch.load(model_path, map_location=device)
        # 兼容训练时的保存格式（优先匹配'params'键）
        if 'params' in ckpt:
            model.load_state_dict(ckpt['params'])
        else:
            model.load_state_dict(ckpt)
        print(f"模型加载成功: {os.path.basename(model_path)}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}") from e

    # 读取图像（保持与数据集处理一致的流程）
    low_img = cv2.imread(low_path)  # 读取为BGR格式
    if low_img is None:
        raise ValueError(f"无法读取图像（可能格式不支持）: {low_path}")
    orig_h, orig_w = low_img.shape[:2]
    print(f"输入图像尺寸: {orig_w}x{orig_h}")

    # 预处理（与LOLv2Dataset保持完全一致）
    # 1. BGR转RGB
    low_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    # 2. 归一化到[0, 1]
    low_norm = low_rgb.astype(np.float32) / 255.0
    # 3. 转为CHW格式并转换为Tensor
    low_tensor = torch.from_numpy(low_norm.transpose(2, 0, 1))  # HWC -> CHW
    # 4. 增加批次维度并移动到设备
    low_tensor = low_tensor.unsqueeze(0).to(device)

    # 推理（关闭梯度计算）
    with torch.no_grad():
        enhanced_tensor = model(low_tensor)
        # 处理模型可能返回的多输出（取最后一个输出）
        if isinstance(enhanced_tensor, list):
            enhanced_tensor = enhanced_tensor[-1]

    # 后处理
    # 1. 移除批次维度并转回CPU
    enhanced_np = enhanced_tensor.squeeze().cpu().numpy()  # 形状: [C, H, W]
    # 2. 转换为HWC格式
    enhanced_np = enhanced_np.transpose(1, 2, 0)  # CHW -> HWC
    # 3. 确保值在[0, 1]范围内（防止溢出）
    enhanced_np = np.clip(enhanced_np, 0.0, 1.0)
    # 4. 调整回原始尺寸（若模型输出尺寸不同）
    if enhanced_np.shape[:2] != (orig_h, orig_w):
        enhanced_np = cv2.resize(enhanced_np, (orig_w, orig_h),
                                interpolation=cv2.INTER_AREA)  # 与数据集resize方式一致
    # 5. 转回BGR格式并转换为uint8
    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    enhanced_bgr = (enhanced_bgr * 255).astype(np.uint8)

    # 保存结果
    cv2.imwrite(save_path, enhanced_bgr)
    print(f"增强图像已保存至: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    # 请根据实际文件位置修改以下路径
    enhance_image(
        low_path="test_images/low_light.jpg",  # 输入低光图像路径
        model_path="checkpoints/laenet_best.pth",  # 模型权重路径
        save_path="results/enhanced.jpg"  # 增强结果保存路径
    )
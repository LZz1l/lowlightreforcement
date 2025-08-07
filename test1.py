import cv2
import numpy as np
import torch  # 新增导入

from models.modules.retinexformer import Retinexformer


def enhance_image(low_path, model_path, save_path):
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 检查文件是否存在
    if not os.path.exists(low_path):
        raise FileNotFoundError(f"低光图像文件不存在: {low_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Retinexformer().eval().to(device)  # 移动到设备

    # 加载模型
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

    # 读取和预处理图像
    low = cv2.imread(low_path)
    if low is None:
        raise ValueError(f"无法读取图像: {low_path}")

    low = low[:, :, ::-1]  # BGR转RGB
    low = cv2.resize(low, (256, 256))
    low_tensor = torch.from_numpy(low).permute(2, 0, 1).float() / 255.0
    low_tensor = low_tensor.unsqueeze(0).to(device)  # 增加批次维度并移动到设备

    with torch.no_grad():
        enhanced = model(low_tensor).squeeze().cpu().numpy()

    enhanced = (enhanced * 255).astype(np.uint8)[:, :, ::-1]  # RGB转BGR
    cv2.imwrite(save_path, enhanced)
    print(f"增强图像已保存至: {save_path}")


# 使用示例
if __name__ == "__main__":
    import os  # 新增导入

    enhance_image(
        'C:/Users/ASUS/OneDrive/Desktop/LOLv2/Real_captured/Test/Low/00690.png',
        'checkpoints/epoch_50.pth',
        'results/enhanced_00690.png'
    )
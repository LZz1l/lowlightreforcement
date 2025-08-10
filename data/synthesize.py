import cv2
import numpy as np
import random


def synthesize_extreme_lowlight(gt_img, scene_type='starry'):
    """
    合成极端低光图像（增强鲁棒性）
    scene_type: 'starry'（星空）/ 'mine'（矿井）
    """
    # 校验场景类型
    valid_scenes = ['starry', 'mine']
    if scene_type not in valid_scenes:
        raise ValueError(f"不支持的场景类型: {scene_type}，可选: {valid_scenes}")

    # 校验输入图像（必须是3通道）
    if gt_img.ndim != 3 or gt_img.shape[2] != 3:
        raise ValueError(f"输入图像必须是3通道RGB，当前形状: {gt_img.shape}")

    # 统一转为uint8 [0,255]处理（无论输入是[0,1]还是[0,255]）
    if gt_img.dtype == np.float32 or gt_img.dtype == np.float64:
        # 假设float类型图像已归一化到[0,1]
        gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)
    elif gt_img.dtype != np.uint8:
        # 其他类型强制转为uint8
        gt_img = gt_img.clip(0, 255).astype(np.uint8)

    h, w = gt_img.shape[:2]

    # 1. 降低亮度（模拟低光）
    low_light_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2HSV)
    brightness = random.uniform(0.02, 0.1)  # 极端低光亮度（2%-10%）
    low_light_img[:, :, 2] = (low_light_img[:, :, 2] * brightness).clip(0, 255).astype(np.uint8)
    low_light_img = cv2.cvtColor(low_light_img, cv2.COLOR_HSV2RGB)

    # 2. 添加噪声（低光图像噪声更明显）
    noise_level = random.uniform(0.01, 0.05)  # 噪声强度
    noise = np.random.normal(0, noise_level * 255, low_light_img.shape).astype(np.int16)
    low_light_img = np.clip(low_light_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. 场景特定增强
    if scene_type == 'starry':
        # 星空场景添加星点
        num_stars = random.randint(50, 200)
        for _ in range(num_stars):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            star_radius = random.randint(1, 3)
            brightness = random.randint(150, 255)
            cv2.circle(low_light_img, (x, y), star_radius, (brightness, brightness, brightness), -1)
    elif scene_type == 'mine':
        # 矿井场景添加均匀噪声和微弱光源
        mine_noise = np.random.normal(0, 5, low_light_img.shape).astype(np.int16)
        low_light_img = np.clip(low_light_img.astype(np.int16) + mine_noise, 0, 255).astype(np.uint8)
        # 添加随机微弱光源
        num_lights = random.randint(5, 15)
        for _ in range(num_lights):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(5, 15)
            brightness = random.randint(30, 80)
            cv2.circle(low_light_img, (x, y), radius, (brightness, brightness, brightness), -1)

    # 转回[0,1] float32格式
    return low_light_img.astype(np.float32) / 255.0
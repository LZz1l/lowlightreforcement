import cv2
import numpy as np
import random


def synthesize_extreme_lowlight(gt_img, scene_type='starry'):
    """
    合成极端低光图像
    scene_type: 'starry'（星空）/ 'mine'（矿井）
    """
    # 转为BGR格式（OpenCV默认）
    if gt_img.shape[-1] == 3 and gt_img.dtype == np.float32:
        gt_img = (gt_img * 255).astype(np.uint8)  # 从[0,1]转为[0,255]
    h, w = gt_img.shape[:2]

    # 1. 降低亮度（模拟低光）
    low_light_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2HSV)
    brightness = random.uniform(0.02, 0.1)  # 极端低光亮度（2%-10%）
    low_light_img[:, :, 2] = low_light_img[:, :, 2] * brightness
    low_light_img = cv2.cvtColor(low_light_img, cv2.COLOR_HSV2RGB)

    # 2. 添加噪声（低光图像噪声更明显）
    noise_level = random.uniform(0.01, 0.05)  # 噪声强度
    noise = np.random.normal(0, noise_level * 255, low_light_img.shape).astype(np.int16)
    low_light_img = np.clip(low_light_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. 场景特定增强（星空场景添加星点）
    if scene_type == 'starry':
        num_stars = random.randint(50, 200)
        for _ in range(num_stars):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            star_radius = random.randint(1, 3)
            brightness = random.randint(150, 255)
            cv2.circle(low_light_img, (x, y), star_radius, (brightness, brightness, brightness), -1)

    # 转回[0,1] float32格式
    return low_light_img.astype(np.float32) / 255.0
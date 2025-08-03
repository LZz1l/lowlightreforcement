import os
import cv2
import torch
import numpy as np
from models.laenet import LAENet
from basicsr.utils import img2tensor, tensor2img


def inference_single_image(model, img_path, save_path):
    """单图推理"""
    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 转为Tensor并添加批次维度
    tensor = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0)
    tensor = tensor.cuda() if torch.cuda.is_available() else tensor

    # 推理
    model.eval()
    with torch.no_grad():
        output_tensor = model(tensor)

    # 转为图像并保存
    output_img = tensor2img(output_tensor.squeeze(), rgb2bgr=True)  # 转为BGR保存
    cv2.imwrite(save_path, output_img)
    print(f'增强结果已保存至: {save_path}')


if __name__ == '__main__':
    # 配置
    model_path = './experiments/LAENet_train/models/latest_net_g.pth'
    img_path = './test_images/dark_image.jpg'  # 输入低光图像路径
    save_path = './test_results/enhanced_image.jpg'  # 输出保存路径

    # 加载模型
    model = LAENet()
    model.load_state_dict(torch.load(model_path)['params'])
    model = model.cuda() if torch.cuda.is_available() else model

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 推理
    inference_single_image(model, img_path, save_path)
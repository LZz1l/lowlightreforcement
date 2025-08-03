import cv2
import numpy as np
from models.retinexformer import Retinexformer


def enhance_image(low_path, model_path, save_path):
    model = Retinexformer().eval()
    model.load_state_dict(torch.load(model_path))

    low = cv2.imread(low_path)[:, :, ::-1]  # BGR转RGB
    low = cv2.resize(low, (256, 256))
    low_tensor = torch.from_numpy(low).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        enhanced = model(low_tensor.unsqueeze(0)).squeeze().cpu().numpy()

    enhanced = (enhanced * 255).astype(np.uint8)[:, :, ::-1]  # RGB转BGR
    cv2.imwrite(save_path, enhanced)


# 使用示例
enhance_image('data/LOLv2/Real_captured/Test/Low/00690.png',
              'checkpoints/epoch_50.pth',
              'results/enhanced_00690.png')
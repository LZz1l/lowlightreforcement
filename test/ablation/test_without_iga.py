import torch
from models.laenet import LAENet
from test.test_metrics import test  # 复用测试逻辑

# 定义移除IGA Block的LAENet变体
class LAENetWithoutIGA(LAENet):
    def __init__(self, **kwargs):
        super(LAENetWithoutIGA, self).__init__(** kwargs)
        # 替换Retinex编码器中的IGA Block为普通卷积
        self.retinex_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # 移除IGA Block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )

if __name__ == '__main__':
    # 加载变体模型并测试
    model = LAENetWithoutIGA()
    model.load_state_dict(torch.load('./experiments/LAENet_ablation_without_iga/models/latest_net_g.pth')['params'])
    # 替换测试函数中的模型（需修改test()函数支持传入模型参数）
    test(custom_model=model)
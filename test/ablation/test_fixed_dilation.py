import torch
from models.modules.hfrm_pro import HFRMPro
from models.laenet import LAENet
from test.test_metrics import test

# 定义固定空洞率的HFRM-Pro变体
class HFRMFixed(HFRMPro):
    def __init__(self, in_channels, out_channels):
        super(HFRMFixed, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_rates=[3, 3, 3]  # 固定空洞率为3
        )
        self.gate = None  # 移除动态门控

# 替换LAENet中的HFRM-Pro为固定版本
class LAENetFixedDilation(LAENet):
    def __init__(self, **kwargs):
        super(LAENetFixedDilation, self).__init__(** kwargs)
        self.wavelet_processor = nn.Sequential(
            HFRMFixed(3, 64),  # 固定空洞率
            HFRMFixed(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

if __name__ == '__main__':
    model = LAENetFixedDilation()
    model.load_state_dict(torch.load('./experiments/LAENet_ablation_fixed_dilation/models/latest_net_g.pth')['params'])
    test(custom_model=model)
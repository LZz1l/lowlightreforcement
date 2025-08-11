import torch
import torch.nn as nn  # 新增导入
from models.modules.hfrm_pro import HFRMPro
from models.laenet import LAENet
from test.test_metrics import test  # 修正导入路径


class HFRMFixed(HFRMPro):
    def __init__(self, in_channels, out_channels):
        super(HFRMFixed, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_rates=[3, 3, 3]
        )
        self.gate = None  # 移除动态门控


class LAENetFixedDilation(LAENet):
    def __init__(self, **kwargs):
        super(LAENetFixedDilation, self).__init__(** kwargs)
        self.wavelet_processor = nn.Sequential(
            HFRMFixed(3, 64),
            HFRMFixed(64, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )


if __name__ == '__main__':
    model = LAENetFixedDilation()
    ckpt = torch.load('./experiments/LAENet_ablation_fixed_dilation/models/latest_net_g.pth')
    model.load_state_dict(ckpt['params'])  # 修正加载格式
    test(custom_model=model)
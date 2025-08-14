import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
class PerceptualLoss(nn.Module):
    """感知损失：基于预训练VGG网络的特征差异损失"""

    def __init__(self,
                 loss_weight=1.0,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.0,
                 norm_img=True):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.norm_img = norm_img  # 是否将输入归一化到[0, 255]

        # 图像归一化参数（ImageNet均值和标准差）
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # 加载预训练VGG网络
        if vgg_type == 'vgg19':
            vgg = models.vgg19(pretrained=True)
            # 选择VGG19的特征层（通常使用这些层的输出作为感知特征）
            self.feature_layers = [2, 7, 12, 21, 30]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        elif vgg_type == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.feature_layers = [2, 7, 12, 19, 26]
        else:
            raise ValueError(f"不支持的VGG类型: {vgg_type}")

        # 构建特征提取器（截取VGG的前半部分）
        self.vgg = nn.Sequential(*list(vgg.features.children())[:max(self.feature_layers) + 1])
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结VGG参数

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            # 输入标准化层
            self.input_norm = nn.BatchNorm2d(3, affine=False)

    def forward(self, x, gt):
        """
        Args:
            x: 生成图像 [B, 3, H, W]，范围通常为[0, 1]
            gt: 真实图像 [B, 3, H, W]，范围通常为[0, 1]
        """
        # 归一化图像到[0, 255]（如果需要）
        if self.norm_img:
            x = x * 255.0
            gt = gt * 255.0

        # 标准化（匹配VGG训练时的输入分布）
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        gt = (gt - self.mean.to(gt.device)) / self.std.to(gt.device)

        if self.use_input_norm:
            x = self.input_norm(x)
            gt = self.input_norm(gt)

        # 提取特征
        x_feats = []
        gt_feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            gt = layer(gt)
            if i in self.feature_layers:
                x_feats.append(x)
                gt_feats.append(gt)

        # 计算感知损失（特征L1差异）
        perceptual_loss = 0.0
        for x_feat, gt_feat in zip(x_feats, gt_feats):
            perceptual_loss += F.l1_loss(x_feat, gt_feat)

        # 计算风格损失（可选，特征gram矩阵差异）
        style_loss = 0.0
        if self.style_weight > 0:
            for x_feat, gt_feat in zip(x_feats, gt_feats):
                x_gram = self._gram_matrix(x_feat)
                gt_gram = self._gram_matrix(gt_feat)
                style_loss += F.l1_loss(x_gram, gt_gram)

        # 总损失
        total_loss = (self.perceptual_weight * perceptual_loss +
                     self.style_weight * style_loss) * self.loss_weight
        return total_loss

    @staticmethod
    def _gram_matrix(x):
        """计算特征图的Gram矩阵（用于风格损失）"""
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        gram = torch.bmm(x, x.transpose(1, 2)) / (c * h * w)  # 归一化
        return gram
# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss





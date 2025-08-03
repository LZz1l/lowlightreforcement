import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from basicsr.train import Trainer
from basicsr.utils.options import parse_options
from models.laenet import LAENet
from models.retinexformer import Retinexformer  # 假设教师模型Retinexformer
from train.losses import DistillationLoss
from data.dataset import LAENetDataset


class DistillTrainer(Trainer):
    """蒸馏训练器，继承BasicSR的Trainer扩展"""

    def __init__(self, opt):
        super(DistillTrainer, self).__init__(opt)
        # 加载教师模型（阶段1：正常光照模型蒸馏）
        self.teacher_model = Retinexformer(pretrained=True)
        self.teacher_model.eval()
        self.teacher_model = self.teacher_model.to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False  # 冻结教师模型

        # 蒸馏损失
        self.distill_loss = DistillationLoss(
            loss_weight=opt['distill']['loss_weight'],
            temperature=opt['distill']['temperature']
        ).to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        # 教师模型输出（用于蒸馏）
        with torch.no_grad():
            self.teacher_feat = self.teacher_model.extract_feat(self.lq)  # 提取教师特征

    def optimize_parameters(self, current_iter):
        # 学生模型前向传播
        self.student_feat = self.net_g.extract_feat(self.lq)  # 需在LAENet中实现extract_feat方法
        self.output = self.net_g(self.lq)

        # 计算损失（原任务损失+蒸馏损失）
        self.loss_total = self.loss(self.output, self.gt)  # 原损失（L1+感知损失）
        self.loss_distill = self.distill_loss(self.student_feat, self.teacher_feat)
        self.loss_total += self.loss_distill

        # 反向传播
        self.optimizer_g.zero_grad()
        self.loss_total.backward()
        self.optimizer_g.step()


def main():
    # 解析蒸馏配置（基于基础配置扩展）
    opt = parse_options(config_path='config/distill_config.yaml', is_train=True)
    # 初始化蒸馏训练器
    distill_trainer = DistillTrainer(opt)
    # 开始蒸馏训练
    distill_trainer.train()


if __name__ == '__main__':
    main()
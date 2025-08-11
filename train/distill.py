import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from basicsr.train import Trainer
from basicsr.utils.options import parse_options
from models.laenet import LAENet
from models.modules.retinexformer import Retinexformer
from train.losses import DistillationLoss
from data.datasets.lolv2_dataset import LOLv2Dataset

class DistillTrainer(Trainer):
    def __init__(self, opt):
        super(DistillTrainer, self).__init__(opt)
        # 加载教师模型
        self.teacher_model = Retinexformer().to(self.device)
        # 加载教师模型权重（假设已预训练）
        teacher_ckpt = torch.load(opt['distill']['teacher_pretrain_path'], map_location=self.device)
        self.teacher_model.load_state_dict(teacher_ckpt['params'])
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 蒸馏损失
        self.distill_loss = DistillationLoss(
            loss_weight=opt['distill']['loss_weight'],
            temperature=opt['distill']['temperature']
        ).to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        # 提取教师模型中间特征（修正：使用教师模型的conv_in输出作为特征）
        with torch.no_grad():
            self.teacher_feat = self.teacher_model.conv_in(self.lq)  # 取第一层卷积特征

    def optimize_parameters(self, current_iter):
        # 学生模型提取特征（取retinex_encoder的第一层输出）
        student_feat = self.net_g.retinex_encoder[0](self.lq)  # LAENet的第一层卷积
        self.output = self.net_g(self.lq)

        # 计算总损失（原任务损失+蒸馏损失）
        task_loss = self.loss(self.output, self.gt)  # 原任务损失（如L1）
        distill_loss = self.distill_loss(student_feat, self.teacher_feat)
        self.loss_total = task_loss + distill_loss

        self.optimizer_g.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()


def main():
    opt = parse_options(config_path='config/distill_config.yaml', is_train=True)
    distill_trainer = DistillTrainer(opt)
    distill_trainer.train()


if __name__ == '__main__':
    main()
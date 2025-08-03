import os
from basicsr.train import Trainer
from basicsr.utils.options import parse_options
from basicsr.utils import set_random_seed


def main():
    # 解析配置文件
    opt = parse_options(config_path='config/config.yaml', is_train=True)
    set_random_seed(opt['train']['manual_seed'])  # 设置随机种子，保证可复现性

    # 初始化训练器（继承BasicSR的Trainer）
    trainer = Trainer(opt)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
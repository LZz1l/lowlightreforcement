import os  # 移至开头统一导入
import torch
from torch import nn

from data.datasets.lolv2_dataset import LOLv2Dataset
from models.laenet import LAENet  # 替换为目标模型LAENet
from torch.utils.data import DataLoader
from train.losses import RetinexPerturbationLoss  # 导入专用损失

# 配置
config = {
    'data_root': 'C:/Users/ASUS/OneDrive/Desktop/LOLv2',
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 50,
    'save_dir': './checkpoints'
}

# 创建保存目录
os.makedirs(config['save_dir'], exist_ok=True)

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LAENet().to(device)  # 实例化LAENet
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = RetinexPerturbationLoss(loss_weight=1.0)  # 使用Retinex专用损失

# 数据加载
train_dataset = LOLv2Dataset(config['data_root'], phase='train', real=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = LOLv2Dataset(config['data_root'], phase='test', real=True)  # 用test代替val
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 训练循环
for epoch in range(config['epochs']):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        low = batch['low'].to(device)
        gt = batch['gt'].to(device)

        # LAENet输出增强图像，同时需要获取L和R用于损失计算
        output = model(low)
        # 从模型中提取L和R（需在LAENet中添加属性存储中间结果）
        L = model.decompose_L(model.retinex_encoder(low))  # 补充获取光照分量
        R = model.decompose_R(model.retinex_encoder(low))  # 补充获取反射分量

        # 计算Retinex损失（输入L、R和原始低光图）
        loss = criterion(L, R, low)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * low.size(0)

    avg_train_loss = train_loss / len(train_dataset)

    # 验证过程
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low = batch['low'].to(device)
            gt = batch['gt'].to(device)
            output = model(low)
            # 验证时用L1损失评估增强效果
            val_loss += nn.L1Loss()(output, gt).item() * low.size(0)

    avg_val_loss = val_loss / len(val_dataset)

    print(f'Epoch {epoch + 1}/{config["epochs"]}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}')

    # 保存模型（统一格式：包含params键，与测试代码匹配）
    torch.save({'params': model.state_dict()}, f'{config["save_dir"]}/epoch_{epoch + 1}.pth')
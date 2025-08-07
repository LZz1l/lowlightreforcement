import torch
from torch import nn

from data.datasets.lolv2_dataset import LOLv2Dataset
from models.modules.retinexformer import Retinexformer
from torch.utils.data import DataLoader

# 配置
config = {
    'data_root': 'C:/Users/ASUS/OneDrive/Desktop/LOLv2',  # 你的绝对路径（注意用正斜杠或双反斜杠）
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 50,
    'save_dir': './checkpoints'
}

# 创建保存目录
import os

os.makedirs(config['save_dir'], exist_ok=True)  # 新增目录创建

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Retinexformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()

# 数据加载
train_dataset = LOLv2Dataset(config['data_root'], phase='train', real=True)  # 修改为小写train
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# 新增验证集加载
val_dataset = LOLv2Dataset(config['data_root'], phase='val', real=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 训练循环
for epoch in range(config['epochs']):
    model.train()
    train_loss = 0.0  # 新增累计损失

    for batch in train_loader:
        low = batch['low'].to(device)
        gt = batch['gt'].to(device)

        pred = model(low)
        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()  # 修复loss_grad错误
        optimizer.step()

        train_loss += loss.item() * low.size(0)  # 累计损失

    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_dataset)

    # 验证过程
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low = batch['low'].to(device)
            gt = batch['gt'].to(device)
            pred = model(low)
            loss = criterion(pred, gt)
            val_loss += loss.item() * low.size(0)

    avg_val_loss = val_loss / len(val_dataset)

    # 更详细的日志输出
    print(f'Epoch {epoch + 1}/{config["epochs"]}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}')

    torch.save(model.state_dict(), f'{config["save_dir"]}/epoch_{epoch + 1}.pth')
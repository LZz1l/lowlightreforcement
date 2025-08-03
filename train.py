import torch
from datasets.lolv2_dataset import LOLv2Dataset
from models.retinexformer import Retinexformer
from torch.utils.data import DataLoader

# 配置
config = {
    'data_root': './data/LOLv2',
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 50,
    'save_dir': './checkpoints'
}

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Retinexformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()

# 数据加载
train_dataset = LOLv2Dataset(config['data_root'], phase='Train', real=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# 训练循环
for epoch in range(config['epochs']):
    model.train()
    for batch in train_loader:
        low = batch['low'].to(device)
        gt = batch['gt'].to(device)

        pred = model(low)
        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss_grad.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), f'{config["save_dir"]}/epoch_{epoch + 1}.pth')
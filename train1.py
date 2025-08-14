import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from data.datasets.lolv2_dataset import LOLv2Dataset
from models.laenet import LAENet
from train.losses import RetinexPerturbationLoss
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练


def main():
    # 配置参数（内存优化）
    config = {
        'data_root': 'C:/Users/ASUS/OneDrive/Desktop/LOLv2',  # 数据集路径
        'batch_size': 2,  # 减小批量大小
        'epochs': 50,
        'lr': 1e-4,
        'img_size': (256, 256),  # 降低分辨率
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    print(f"使用设备: {config['device']}")

    # 1. 加载数据集（添加缩放）
    train_dataset = LOLv2Dataset(
        config['data_root'],
        phase='train',
        real=True,
        resize=config['img_size']
    )
    val_dataset = LOLv2Dataset(
        config['data_root'],
        phase='test',  # 使用test作为验证集
        real=True,
        resize=config['img_size']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型、损失函数、优化器
    model = LAENet(base_channels=32).to(config['device'])  # 减小通道数
    criterion = RetinexPerturbationLoss(loss_weight=1.0).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 3. 混合精度训练配置
    scaler = GradScaler()

    # 4. 训练循环
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0

        # 训练
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            low = batch['low'].to(config['device'])
            gt = batch['gt'].to(config['device'])

            optimizer.zero_grad()

            # 启用混合精度
            with autocast():
                output = model(low)
                # 分解光照和反射分量
                retinex_feat = model.retinex_encoder(low)
                L = model.decompose_L(retinex_feat)
                R = model.decompose_R(retinex_feat)
                loss = criterion(L, R, low)  # 使用Retinex损失

            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * low.size(0)

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}, 训练损失: {avg_train_loss:.4f}")

        # 验证（简化版）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                low = batch['low'].to(config['device'])
                with autocast():
                    retinex_feat = model.retinex_encoder(low)
                    L = model.decompose_L(retinex_feat)
                    R = model.decompose_R(retinex_feat)
                    loss = criterion(L, R, low)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch + 1}, 验证损失: {avg_val_loss:.4f}\n")

        # 保存模型
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    main()

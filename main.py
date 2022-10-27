import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models.srresnet import SRResNet
from datasets.SRdataset import SRDataset
from utils.averagemeter import AverageMeter
 
 
# 数据集参数
H_images = './data/DIV2K/DIV2K_train_HR'
L_images = './data/DIV2K/DIV2K_train_LR_bicubic/X4'
crop_size = 96      # 高分辨率图像裁剪尺寸

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
 
# 学习参数
checkpoint = None   # 预训练模型路径，如果不存在则为None
batch_size = 128    # 批大小
epochs = 200        # 迭代轮数
workers = 0         # 工作线程数
lr = 1e-4           # 学习率
 
# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True # 对卷积进行加速
 
writer = SummaryWriter() # 实时监控     使用命令 tensorboard --logdir runs  进行查看
 
def main():
    """
    训练.
    """
    global checkpoint, writer
 
    # 初始化
    model = SRResNet()

    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
 
    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
 
    # 加载预训练模型
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    # 定制化的dataloaders
    train_dataset = SRDataset(H_images, L_images, split='train', crop_size=crop_size, scaler=4)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                pin_memory=True) 
 
    # 开始逐轮训练
    for epoch in range(0, epochs+1):
 
        model.train()  # 训练模式：允许使用批样本归一化
        loss_epoch = AverageMeter()  # 统计损失函数
        n_iter = len(train_loader)
        total_loss = 0.0
        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(tqdm(train_loader)):
 
            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  [-1, 1]格式
 
            # 前向传播
            sr_imgs = model(lr_imgs)
 
            # 计算损失
            loss = criterion(sr_imgs, hr_imgs)  
            total_loss += loss.item()
            # 后向传播
            optimizer.zero_grad()
            loss.backward()
 
            # 更新模型
            optimizer.step()
 
            # 记录损失值
            loss_epoch.update(loss.item(), lr_imgs.size(0))
 
            # 监控图像变化
            if i==(n_iter-2):
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                writer.add_image('SRResNet/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
 
        # 手动释放内存              
        del lr_imgs, hr_imgs, sr_imgs
 
        # 监控损失值变化
        writer.add_scalar('SRResNet/MSE_Loss', total_loss, epoch)    
 
        # 保存预训练模型
        torch.save(model.state_dict(), 'checkpoints/checkpoint_srresnet_' + str(epoch) + '.pt')
    
    # 训练结束关闭监控
    writer.close()
 
 
if __name__ == '__main__':
    main()
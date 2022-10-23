import torch.backends.cudnn as cudnn
import torch
import tqdm as tqdm
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from datasets.SRdataset import SRDataset
from utils.averagemeter import AverageMeter

train_dataset = SRDataset(split='train', crop_size=96)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=128,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True) 

for epoch in range(0, 10):
    loss_epoch = AverageMeter()  # 统计损失函数
    n_iter = len(train_loader)
    total_loss = 0.0
    # 按批处理
    for i, (lr_imgs, hr_imgs) in enumerate(tqdm(train_loader)):

        # 数据移至默认设备进行训练
        lr_imgs = lr_imgs
        hr_imgs = hr_imgs
 
import torch
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from datasets.SRdataset import SRDataset
from utils.averagemeter import AverageMeter

train_dataset = SRDataset(split='train', crop_size=96)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=128,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True) 

for epoch in range(0, 10):
    loss_epoch = AverageMeter()  # 统计损失函数
    n_iter = len(train_loader)
    total_loss = 0.0
    # 按批处理
    for i, (lr_imgs, hr_imgs) in enumerate(tqdm(train_loader)):

        # 数据移至默认设备进行训练
        _imgs = transforms.ToPILImage()(lr_imgs[0])
        _imgs.show()
        r_imgs = transforms.ToPILImage()(hr_imgs[0])
        r_imgs.show()
 
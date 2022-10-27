from re import L
from torch import nn
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets.SRdataset import SRDataset
from models.SRmodel import SRResNet
from utils.averagemeter import AverageMeter
import time
 
# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
ngpu = 2                # GP数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def tensor_to_np(tensor):
    img = tensor.mul(255).byte() # 取值范围
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0)) # 改变数据大小
    return img

if __name__ == '__main__':
    
    # 测试集目录
    test_data_names = ["Set5+14", "urban100"]
 
    # 预训练模型
    srresnet_checkpoint = "./checkpoints/checkpoint_srresnet.pth"
 
    # 加载模型SRResNet
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(torch.load(srresnet_checkpoint))
   
    srresnet.eval()
    model = srresnet


    # 定制化数据加载器
    for H_images in test_data_names:
        if H_images == 'Set5+14':
            test_dataset = SRDataset(H_images='./data/test/Set5+14/original', L_images='./data/test/Set5+14/LRbicx4', split='test', crop_size=0, scaler=4)
        else:
            test_dataset = SRDataset(H_images = './data/test/urban100', split='test', crop_size=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        # 记录每个样本 PSNR 和 SSIM值
        PSNRs = 0.0
        SSIMs = 0.0

        # 记录测试时间
        start = time.time()

        with torch.no_grad():
            # 逐批样本进行推理计算
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                
                # 数据移至默认设备
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # 前向传播.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]                

                # 计算 PSNR 和 SSIM
                # sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
                # hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel

                psnr = peak_signal_noise_ratio(tensor_to_np(hr_imgs), tensor_to_np(sr_imgs), data_range=255.)
                ssim = structural_similarity(tensor_to_np(hr_imgs), tensor_to_np(sr_imgs), data_range=255.)
                PSNRs += psnr
                SSIMs += ssim


        # 输出平均PSNR和SSIM
        print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
        print('平均单张样本用时  {:.3f} 秒'.format((time.time()-start)/len(test_dataset)))
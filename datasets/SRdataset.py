import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import Transform
 
 
class SRDataset(Dataset):
    """
    数据集加载器
    """
 
    def __init__(self, split, crop_size, scaling_factor, test_data_name=None):
        """
        :参数 data_folder: # Json数据文件所在文件夹路径
        :参数 split: 'train' 或者 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸  （实际训练时不会用原图进行放大，而是截取原图的一个子块进行放大）
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        :参数 test_data_name: 如果是评估阶段，则需要给出具体的待评估数据集名称，例如 "Set14"
        """
 
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)

        self.test_data_name = test_data_name
 
        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("请提供测试数据集名称!")

 
        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        # 如果是测试，则不需要对图像的长宽作限定
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "裁剪尺寸不能被放大比例整除!"
 
        # 读取图像路径
        if self.split == 'train':
            # with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
            #     self.images = json.load(j)
            self.Himages = os.listdir('./data/DIV2K/DIV2K_train_HR')
            self.Limages = os.listdir('./data/DIV2K/DIV2K_train_LR_bicubic/X4')
        else:
            # with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
            #     self.images = json.load(j)
            self.Limages = os.listdir('./data/DIV2K/DIV2K_test_LR_bicubic/X4')

        # 数据处理方式
        self.transform = Transform(split=self.split,
                                         crop_size=self.crop_size)
 
    def __getitem__(self, i):
        """
        为了使用PyTorch的DataLoader, 必须提供该方法.
        :参数 i: 图像检索号
        :返回: 返回第i个低分辨率和高分辨率的图像对
        """
        # 读取图像
        H_img = Image.open(self.Himages[i], mode='r').convert('RGB')
        L_img = Image.open(self.Limages[i], mode='r').convert('RGB')
        L_img, H_img = self.transform(L_img), self.transform(H_img)
        return L_img, H_img
 
    def __len__(self):
        """
        为了使用PyTorch的DataLoader, 必须提供该方法.
        :返回: 加载的图像总数
        """
        return len(self.images)
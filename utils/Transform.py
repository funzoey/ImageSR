import random
from PIL import Image

class ImageTransforms(object):
    """
    图像变换.
    """
 
    def __init__(self, split, crop_size, scaling_factor):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
 
        assert self.split in {'train', 'test'}
 
    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """
 
        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
 
        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)
 
        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor
 
        return lr_img, hr_img
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import random

class ImageTransforms(object):
    """
    图像变换.
    """
 
    def __init__(self, scaler):
        """
        :参数 crop_size: 高分辨率图像裁剪尺寸
        """
        self.scaler =scaler
        self.trans = transforms.Compose([
            # transforms.CenterCrop ,
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
 
    def __call__(self, hr_img, lr_img = None, crop_size = 0):
        if lr_img == None:
            lr_img = hr_img.resize((int(hr_img.width / self.scaler), int(hr_img.height / self.scaler)), Image.BICUBIC)
            
        # 裁剪
        if crop_size != 0:
            left = random.randint(0, lr_img.width - crop_size)
            top = random.randint(0, lr_img.height - crop_size)
            lr_img = F.crop(lr_img, top, left, crop_size, crop_size)
            hr_img = F.crop(hr_img, top * self.scaler, left * self.scaler, crop_size * self.scaler, crop_size * self.scaler)

        return self.trans(lr_img), self.trans(hr_img)
        
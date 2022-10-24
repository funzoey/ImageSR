from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


class ImageTransforms(object):
    """
    图像变换.
    """
 
    def __init__(self, split, scaler):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        """
        self.split = split.lower()
        self.scaler =scaler
        self.trans = transforms.Compose([
            # transforms.CenterCrop ,
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        assert self.split in {'train', 'test'}
 
    def __call__(self, img, crop_size = 96, top = 0, left = 0):

        # 裁剪
        if self.split == 'train':
            _img = F.crop(img, top, left, crop_size, crop_size)
            return self.trans(_img)

        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaler + 1
            y_remainder = img.height % self.scaler + 1
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))
            lr_img = hr_img.resize((hr_img.width / self.scaler, hr_img.height / self.scaler), Image.BICUBIC)
            return lr_img, hr_img
        
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

class ImageTransforms(object):
    """
    图像变换.
    """
 
    def __init__(self, split, crop_size):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.trans = transforms.Compose([
            # transforms.CenterCrop ,
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        assert self.split in {'train', 'test'}
 
    def __call__(self, img, top = 0, left = 0):

        # 裁剪
        if self.split == 'train':
            _img = F.crop(img, top, left, self.crop_size, self.crop_size)

        # else:
        #     # 从图像中尽可能大的裁剪出能被放大比例整除的图像
        #     x_remainder = img.width % self.scaling_factor
        #     y_remainder = img.height % self.scaling_factor
        #     left = x_remainder // 2
        #     top = y_remainder // 2
        #     right = left + (img.width - x_remainder)
        #     bottom = top + (img.height - y_remainder)
        #     hr_img = img.crop((left, top, right, bottom))
 
        return self.trans(_img)
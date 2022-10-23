import random
from PIL import Image

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
 
        assert self.split in {'train', 'test'}
 
    def __call__(self, img):

        # 裁剪
        if self.split != None:
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            _img = img.crop((left, top, right, bottom))

        # else:
        #     # 从图像中尽可能大的裁剪出能被放大比例整除的图像
        #     x_remainder = img.width % self.scaling_factor
        #     y_remainder = img.height % self.scaling_factor
        #     left = x_remainder // 2
        #     top = y_remainder // 2
        #     right = left + (img.width - x_remainder)
        #     bottom = top + (img.height - y_remainder)
        #     hr_img = img.crop((left, top, right, bottom))
 
        return _img
from typing import Any
from torchvision import transforms
from utils.rmbg_postprocess import MaskPostProcessor
import numpy as np

postprocessor = MaskPostProcessor()

def mask_post_process_pp(mask):

    mask_processed = postprocessor.apply_3_sigmoid_region_process(mask)
    mask_processed = postprocessor.apply_erode(mask_processed, 3)

    return mask_processed

def mask_post_process_bg(mask):

    mask_processed = postprocessor.apply_3_sigmoid_region_process(mask)
    mask_processed = postprocessor.apply_dilate(mask_processed, 2)

    return np.ones_like(mask_processed) * 255 - mask_processed


class ResizeNormalizeTransformer:
    def __init__(self, height=128, width=128, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.image = None
        self.resize_normalize = \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize(mean=mean, std=std)
            ])
        
    def __call__(self, image):
        self.image = self.resize_normalize(image)
        return self.image

class MaskCropTransformer:

    def __init__(self, height=128, width=128, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        # check target __call__ parameter
        self.image = None
        self.mask = None
        
        self.resize_normalize = \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        
    def make_mask(self, mask):
        mask_pp = mask_post_process_pp(mask)
        bin_mask = np.where(mask_pp < 127, 0, 1)
        return bin_mask
    
    def get_crop_image(self, image):

        y, x = np.where(image[:,:,0] != 0)

        # 바운딩 박스 계산
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # 이미지 크롭
        cropped_image = image[y_min:y_max+1, x_min:x_max+1,:]

        return cropped_image.astype(np.uint8)

    def __call__(self, image, mask):
        
        image = np.clip(image + 1, 1, 255)
        mask = self.make_mask(mask)

        masked_crop_image = self.get_crop_image(image * mask)
        masked_crop_image = self.resize_normalize(masked_crop_image)
        
        return masked_crop_image

import cv2
if __name__ == "__main__":
    from PIL import Image
    img = np.array(Image.open("/media/mlfavorfit/sda/template_recommend_dataset/train_origin/alcohol/1289_perso.jpg"))
    small_img = cv2.resize(img,[128,128])
    print(small_img.shape)
    # Image.fromarray(small_img).show()

    t = ResizeNormalizeTransformer()
    a = np.array(t(img))

    # Normalize에서 사용된 평균과 표준편차
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    colors = (a*std.reshape(1,-1,1,1) + mean.reshape(1,-1,1,1))*255
    color = colors[0].astype(np.uint8)

    print(Image.fromarray(color.transpose(1,2,0)).show())
# 배경 이미지의 광원 방향과 세기를 분석하여
# 오브젝트 이미지에 그림자를 생성합니다.

import numpy as np
import cv2
import base64
from io import BytesIO
import pyheif
from PIL import Image

def bs64_to_arr(bs64_str):
    bs64_str_splited = bs64_str.split(',')[1]
    decoded_bytes = base64.b64decode(bs64_str_splited)
    buffer = BytesIO(decoded_bytes)

    if bs64_str.startswith("data:img/png;base64"):
        img_arr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    elif bs64_str.startswith("data:image/heic;base64"):
        img_arr = heif_to_arr(decoded_bytes)
    else:
        img_arr = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    return img_arr

def arr_to_bs64(img_arr):
    _, buffer = cv2.imencode('.png', img_arr)
    bs64_str = base64.b64encode(buffer).decode('utf-8')
    return bs64_str

def heif_to_arr(heif_bytes):
    heif_file = pyheif.read(heif_bytes)
    
    image_pil = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image_arr = np.array(image_pil).astype(np.uint8)
    
    return image_arr

def custom_func(x):
    if x > 1200:
        x = 1200
    elif x < 200:
        x = 200
    # return -0.00027*(x - 700) + (0.3 + 0.03)/2
    # return -0.00018*x + 0.236
    # return -0.000135*x + 0.177
    return -0.00009*x + 0.118

def apply_gaussian_blur(img, magnitude):

    ratio = 0.03/500 * (700 - magnitude) + 0.05
    kernel_size = int(min(img.shape[:2]) * ratio)

    if kernel_size % 2 == 0:
        kernel_size += 1

    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img


class ShadowMaker:
    def __init__(self):
        pass
    

    def extract_light_degree_and_magnitude(self, img_arr):
        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

        # Sobel 연산자를 사용하여 x, y 방향의 그라디언트 계산
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # 그라디언트의 방향과 크기 계산
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        # 가장 큰 그라디언트 크기를 가진 픽셀의 방향을 반환
        degree = angle[np.unravel_index(np.argmax(magnitude), magnitude.shape)]
        magnitude = np.max(magnitude)

        return degree, magnitude
    

    def get_bottom_center(self, shadow_arr):
        _, binary_mask = cv2.threshold(shadow_arr[:,:,0], 0, 1, cv2.THRESH_OTSU)
        for row in range(binary_mask.shape[0]-1, 0, -1):
            if np.sum(binary_mask[row]) != 0:
                obj_pixel = np.where(binary_mask[row]==1)[0]
                return (row - (1/100*shadow_arr.shape[0]), int((obj_pixel[0] + obj_pixel[-1])/2))
        return (None, None)
    

    def linear_transform_topdown(self, shadow_arr, degree, magnitude):

        # 변환 행렬 계산
        shadow_dx = 1/magnitude * (15*shadow_arr.shape[1]) * np.cos(np.radians(degree))
        shadow_dy = 1/magnitude * (15*shadow_arr.shape[0]) * np.sin(np.radians(degree))

        M = np.float32([[1, 0, shadow_dx], [0, 1, shadow_dy]])
        
        # 선형 변환을 사용하여 그림자 생성
        shadow_arr = cv2.warpAffine(shadow_arr, M, (shadow_arr.shape[1], shadow_arr.shape[0]))

        return shadow_arr, M
    

    def get_crop_mask(self, object_mask):
        _, binary_mask = cv2.threshold(object_mask[:,:,0], 0, 1, cv2.THRESH_OTSU)

        y, x = np.where(binary_mask == 1)
        
        # 바운딩 박스 계산
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # 마스크 크롭
        cropped_mask = object_mask[y_min:y_max+1, x_min:x_max+1,:]

        return cropped_mask
    
    def get_crop_image(self, image):

        y, x = np.where(image[:,:,0] != 0)

        # 바운딩 박스 계산
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # 이미지 크롭
        cropped_image = image[y_min:y_max+1, x_min:x_max+1,:]

        return cropped_image.astype(np.uint8)
    

    def linear_transform_floor(self, shadow_arr, degree, magnitude):
    
        cropped_mask = self.get_crop_mask(shadow_arr)
        c_y_crop, c_x_crop = self.get_bottom_center(cropped_mask)
        c_y, c_x = self.get_bottom_center(shadow_arr)

        # 중심을 원점으로 이동하는 변환 행렬
        translation_matrix = np.float32([
            [1, 0, -c_x_crop],
            [0, 1, -c_y_crop],
            [0, 0, 1]
        ])

        # 중심을 원래 위치로 되돌리는 변환 행렬
        inv_translation_matrix = np.float32([
            [1, 0, c_x],
            [0, 1, c_y],
            [0, 0, 1]
        ])

        tilt = - 1/180 * np.abs(degree-180) + 0.5
        y_scale = custom_func(magnitude)

        transform_marix = np.float32([[1,tilt,0],[0,y_scale,0],[0,0,1]])
        M = np.dot(inv_translation_matrix, np.dot(transform_marix, translation_matrix))[:2, :]

        # 선형 변환을 사용하여 그림자 생성
        shadow_arr = cv2.warpAffine(cropped_mask, M, (shadow_arr.shape[1], shadow_arr.shape[0]))

        return shadow_arr, M


    def get_blur_value(self, magnitude):
        if magnitude > 1100:
            magnitude = 1100
        elif magnitude < 300:
            magnitude = 300
        # return -0.06*magnitude + 118
        return -0.0666667*magnitude + 100

    def get_transparency_value(self, magnitude):
        if magnitude > 1100:
            magnitude = 1100
        elif magnitude < 300:
            magnitude = 300
        return 0.0875*magnitude - 3.75

    def mask_2_blackobj(self, mask_img):

        h, w = mask_img.shape[:2]
        black_obj = np.zeros((h, w, 4), dtype=mask_img.dtype)

        black_obj[:,:,3] = mask_img[:,:,0]

        return black_obj


    def process(self, template_arr, shadow_arr, angle="topdown"):
        degree, magnitude = self.extract_light_degree_and_magnitude(template_arr)
       
        if shadow_arr.shape[2] == 4: 
            shadow_arr = np.repeat(shadow_arr[:,:,3][:,:,np.newaxis], 3, axis=2)    

        if angle == "topdown":
            shadow_arr, M = self.linear_transform_topdown(shadow_arr, degree, magnitude)
        elif angle == "floor":
            shadow_arr, M = self.linear_transform_floor(shadow_arr, degree, magnitude)

        blur_value = self.get_blur_value(magnitude)
        transparency_value = self.get_transparency_value(magnitude)

        black_shadow_png = self.mask_2_blackobj(shadow_arr)

        bs64_shadow = arr_to_bs64(black_shadow_png)

        # return {"shadow":black_shadow_png, "blur":blur_value, "opacity":transparency_value, "matrix":M} # test code
        return {"shadow":bs64_shadow, "blur":blur_value, "opacity":transparency_value, "matrix":M}
    

if __name__ == "__main__":
    shadow_img_arr = None
    template_img_arr = None
    
    shadowmaker = ShadowMaker()

    '''
    result json key: 
        shadow = base64 image, blur = blur value(0~100), 
        opacity = transparency value(0~100), M = lenear transform matrix(Affine)
    '''

    result_json = shadowmaker.process(template_img_arr, shadow_img_arr, angle="topdown")
    # result_json = shadowmaker.process(template_img_arr, shadow_img_arr, angle="floor")
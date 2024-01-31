import torch
# from model.res_cnn import CNNEncoder
from model.cnn_encoder import CNNEncoder
from model.convnext import ConvnextDecoder
from model.swin import SwinbDecoder

from transform import MaskCropTransformer
from PIL import Image
import numpy as np

def to_color_percentage(arr):

    arr = arr.reshape(1,4,4)
    colors = arr[:,:,:3]
    percentage = arr[:,:,3]

    return colors.tolist(), percentage.tolist()

state_path = "./ckpt/ckpt_40.pth"
# model = ConvnextDecoder(16,[512])
# model = SwinbDecoder()
model = CNNEncoder(in_channels=3, out_channels=16, hidden_dims=[32,64,128,256], num_inter_layers=3)

model.load_state_dict(torch.load(state_path, map_location="cpu"))

# fn = "contraceptive/67_clear"
# fn = "contraceptive/82_vitam"
# fn = "beer/355_South"
fn = "bottle/2640_abrow"
image_path = f"/media/mlfavorfit/sda/template_recommend_dataset/train_origin/{fn}.jpg"
mask_path = f"/media/mlfavorfit/sda/template_recommend_dataset/train_origin/{fn}_mask.jpg"

transformer = MaskCropTransformer(224, 224)

image = np.array(Image.open(image_path).convert("RGB"))
mask = np.array(Image.open(mask_path).convert("RGB"))


input_data = transformer(image, mask)

model.eval()
with torch.no_grad():
    output = model(input_data.unsqueeze(0))

# 예측 색상
colors, percenatage = to_color_percentage(output)
print(colors, percenatage)

from utils.colors_utils import color_normalization, color_normalization_restore, rgb2hsv_cv2, visualize_rgb_colors, hsv2rgb_cv2
restored_colors = color_normalization_restore(colors, scaling=True, type="rgb")
# restored_colors = color_normalization_restore(colors, scaling=True, type="hsv")
visualize_rgb_colors(np.clip(restored_colors[0],0,255))


# 진짜 색상
import json

with open("/media/mlfavorfit/sda/template_recommend_dataset/cluster_features_split/train_labels.json", mode="r") as f:
    temp = json.load(f)
colors, percenatage = temp[image_path.replace("/train/", "/train_origin/")]
visualize_rgb_colors(np.array(colors))

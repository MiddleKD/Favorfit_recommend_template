import torch
from model.reversed_encoder import ReversedAutoEncoder, ReversedVAE
from model.simple_resnet import SimpleNNWithResBlocks

import numpy as np

def to_color_percentage(arr):

    arr = arr.reshape(1,4,4)
    colors = arr[:,:,:3]
    percentage = arr[:,:,3]

    return colors.tolist(), percentage.tolist()

state_path = "./ckpt/ckpt_10.pth"
# model = ReversedAutoEncoder(16, 16, 3, [32,64,128])
# model = SimpleNNWithResBlocks()
# model = ReversedAutoEncoder(16, 16)
model = ReversedVAE(16,16,256,3,[32,64,128])

model.load_state_dict(torch.load(state_path, map_location="cpu"))


# fn = "contraceptive/82_vitam"
# fn = "beerglass/1272_clear"
fn = "beer/355_South"
# fn = "bottle/2640_abrow"
image_path = f"/media/mlfavorfit/sda/template_recommend_dataset/train_origin/{fn}.jpg"

import json

with open("/media/mlfavorfit/sda/template_recommend_dataset/cluster_features_split/train_datas.json", mode="r") as f:
    datas = json.load(f)

import numpy as np

from utils.colors_utils import color_normalization, color_normalization_restore, rgb2hsv_cv2, visualize_rgb_colors, hsv2rgb_cv2
def sort_data(data):
    colors, weights = data
    sorted_label = sorted(list(zip(colors, weights)), key = lambda x:x[1], reverse=True)
    
    colors_mat = []
    weights_mat = []
    for color, weight in sorted_label:
        # hsv_color = rgb2hsv_cv2(np.array(color))
        # colors_mat.append(color_normalization(hsv_color, scaling= True, type="hsv"))
        # colors_mat.append(color_normalization(color, scaling= True, type="rgb"))
        colors_mat.append(np.array(color).astype(np.float32)/100)
        weights_mat.append(weight)
    
    return torch.cat([torch.FloatTensor(colors_mat), torch.FloatTensor(weights_mat).reshape(4,1)], dim=1)

input_data = sort_data(datas[image_path]).flatten()

model.eval()
with torch.no_grad():
    output = model(input_data.unsqueeze(0))[0]

colors, percentage = to_color_percentage(output)
print(colors, percentage)

# restored_colors = color_normalization_restore(colors, scaling=True, type="hsv")
# visualize_rgb_colors(hsv2rgb_cv2(restored_colors))
restored_colors = color_normalization_restore(colors, scaling=True, type="rgb")

print(restored_colors)
visualize_rgb_colors(np.clip(restored_colors[0], 0, 255))

with open("/media/mlfavorfit/sda/template_recommend_dataset/cluster_features_split/train_labels.json", mode="r") as f:
    datas = json.load(f)

colors, percentage = datas[image_path]
visualize_rgb_colors(colors)
import numpy as np
import json
import os
from models.model import forward
from utils.colors_utils import color_normalization, color_normalization_restore

# JSON 파일에서 모델 가중치 로드
relative_path = '202311162115.json'
absolute_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)
with open(absolute_path, 'r') as rf:
    model_weights = json.load(rf)

# 가중치와 편향을 NumPy 배열로 변환
for key in model_weights:
    model_weights[key] = np.array(model_weights[key])

def to_color_percentage(arr):
    
    arr = arr.reshape(-1,4,4)
    colors = arr[:,:,:3]
    percentage = arr[:,:,3]
    
    return colors, percentage

def sort_data(data):
    colors, weights = data
    sorted_label = sorted(list(zip(colors, weights)), key = lambda x:x[1], reverse=True)
    
    colors_mat = []
    weights_mat = []
    for color, weight in sorted_label:
        colors_mat.append(color_normalization(color, scaling= True, type="rgb"))
        weights_mat.append(weight)
    
    return np.concatenate([np.array(colors_mat, dtype=np.float32), np.array(weights_mat, dtype=np.float32).reshape(4,1)], axis=1)

def run(colors_weights_pair):

    input = sort_data(colors_weights_pair).flatten()

    out = forward(input, model_weights)
    
    colors, percentage = to_color_percentage(out)
    restored_colors = color_normalization_restore(colors, scaling=True, type="rgb")
    
    return np.clip(restored_colors, 0, 255), np.clip(percentage, 0, 1)

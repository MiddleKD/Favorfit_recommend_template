import torch
from models.torch_model import LinearRes
from utils.colors_utils import color_normalization, color_normalization_restore

def to_color_percentage(arr):
    
    arr = arr.reshape(-1,4,4)
    colors = arr[:,:,:3]
    percentage = arr[:,:,3]
    
    return colors.numpy(), percentage.numpy()

def sort_data(data):
    colors, weights = data
    sorted_label = sorted(list(zip(colors, weights)), key = lambda x:x[1], reverse=True)
    
    colors_mat = []
    weights_mat = []
    for color, weight in sorted_label:
        colors_mat.append(color_normalization(color, scaling= True, type="rgb"))
        weights_mat.append(weight)
    
    return torch.cat([torch.FloatTensor(colors_mat), torch.FloatTensor(weights_mat).reshape(4,1)], dim=1)

def run(colors_weights_pair):
    state_path = "./models/202311162115.pth"

    model = LinearRes(input_size=16, output_size=16, inter_num=3, expand_dims=[32, 64, 128])
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    
    input = sort_data(colors_weights_pair).flatten()

    model.eval()
    with torch.no_grad():
        out = model(input.unsqueeze(0))[0]
    
    colors, percentage = to_color_percentage(out)
    restored_colors = color_normalization_restore(colors, scaling=True, type="rgb")

    return restored_colors, percentage

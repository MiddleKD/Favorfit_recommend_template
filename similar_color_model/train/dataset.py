from torch.utils.data import Dataset
import torch
from PIL import Image
from glob import glob
import json, os
import numpy as np
from utils.colors_utils import color_normalization, color_normalization_restore, rgb2hsv_cv2, visualize_rgb_colors, hsv2rgb_cv2

class ProductColorRecommendDataset(Dataset):
    def __init__(self, data_dirs, label_path, data_num, transformer = None, null_label = torch.zeros([16])):
        self.null_label = null_label
        fns = []
        for data_dir in data_dirs:
            fns.extend(glob(data_dir)) 
        fns = sorted(fns)

        with open(os.path.join(label_path), mode="r") as f:
            label_map = json.load(f)

        self.transformer = transformer
        self.datas = [(img_path,
                       self.make_sort_label(label_map[img_path.replace("/train/", "/train_origin/").replace("/val/", "/val_origin/")])) 
                       for img_path in fns[:data_num]]

        print(f"Dataset -> data_path:{data_dir}, num_of_data:{len(self.datas)}")


    def make_sort_label(self, label):
        if len(label) == 1: return self.null_label

        colors, weights = label
        sorted_label = sorted(list(zip(colors, weights)), key = lambda x:x[1], reverse=True)
        
        colors_mat = []
        weights_mat = []
        for color, weight in sorted_label:
            # hsv_color = rgb2hsv_cv2(np.array(color))
            # colors_mat.append(color_normalization(hsv_color, scaling= True, type="hsv"))
            colors_mat.append(color_normalization(color, scaling= True, type="rgb"))
            weights_mat.append(weight)
        
        return torch.cat([torch.FloatTensor(colors_mat), torch.FloatTensor(weights_mat).reshape(4,1)], dim=1)
    
    def __getitem__(self, idx):
        image_path, label = self.datas[idx]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)

        if hasattr(self.transformer, "mask"):
            image = self.transformer(image)
        else:
            image = self.transformer(image)

        return image, label.flatten()

    def __len__(self):
        return len(self.datas)
    

class ClusterColorRecommendDataset(Dataset):
    def __init__(self, data_json_path, label_json_path, data_num, null_label = torch.zeros([16])):
        self.null_label = null_label
        data_json_path = data_json_path[0]

        with open(os.path.join(data_json_path), mode="r") as f:
            data_map = json.load(f)

        with open(os.path.join(label_json_path), mode="r") as f:
            label_map = json.load(f)
        
        map_keys = sorted(label_map.keys())[:data_num]

        self.datas = [
                        (key,
                         self.sort_data(data_map[key]), 
                         self.sort_data(label_map[key])) 
                        for key in map_keys
                    ]


        print(f"Dataset -> data_path:{data_json_path}, num_of_data:{len(self.datas)}")


    def sort_data(self, data):
        if len(data) == 1: return self.null_label

        colors, weights = data
        sorted_label = sorted(list(zip(colors, weights)), key = lambda x:x[1], reverse=True)
        
        colors_mat = []
        weights_mat = []
        for color, weight in sorted_label:
            # hsv_color = rgb2hsv_cv2(np.array(color))
            # colors_mat.append(color_normalization(hsv_color, scaling= True, type="hsv"))
            colors_mat.append(color_normalization(color, scaling= True, type="rgb"))
            weights_mat.append(weight)
        
        return torch.cat([torch.FloatTensor(colors_mat), torch.FloatTensor(weights_mat).reshape(4,1)], dim=1)
    
    def __getitem__(self, idx):
        key, data, label = self.datas[idx]
        return data.flatten(), label.flatten()

    def __len__(self):
        return len(self.datas)


if __name__ == "__main__":
    from transform import ResizeNormalizeTransformer

    transformer = ResizeNormalizeTransformer(128,128, mean = [0.314 , 0.3064, 0.553], std = [0.2173, 0.2056, 0.2211])
    train_dataset = ProductColorRecommendDataset(["/media/mlfavorfit/sda/template_recommend_dataset/train/*/*"], "/media/mlfavorfit/sda/template_recommend_dataset/train_label.json", data_num=70000, transformer=transformer)
    

    hsv_color = train_dataset[50][1].reshape(-1,4,4)[:,:,:3]

    restored_hsv_color = color_normalization_restore(hsv_color, scaling=True, type="hsv")
    rgb_color = hsv2rgb_cv2(restored_hsv_color)
    
    print(train_dataset[50][1])
    visualize_rgb_colors(rgb_color)

    train_dataset = ClusterColorRecommendDataset(["/media/mlfavorfit/sda/template_recommend_dataset/cluster_features_base/val_datas.json"], "/media/mlfavorfit/sda/template_recommend_dataset/cluster_features_base/val_labels.json", data_num=70000)
    print(train_dataset[0])

import sys, os
current_dir = os.path.dirname(os.path.realpath("__file__"))
sys.path.append(os.path.join(current_dir, '.'))
sys.path.append(os.path.join(current_dir, '..'))

import numpy as np
from PIL import Image

from models.inference import run
from utils.colors_utils import color_extraction
from utils.data_utils import load_templates_features, calculate_cos_similarity, get_close_index, concat_array

def get_datas_with_mask(root_dir):

    fns = sorted(os.listdir(root_dir))

    image_path = []
    mask_path = []
    for fn in fns:
        if fn.endswith("mask.jpg"):
            mask_path.append(os.path.join(root_dir, fn))
        else:
            image_path.append(os.path.join(root_dir, fn))

    image_mask_pair = list(zip(image_path, mask_path))
    return image_mask_pair


from tqdm import tqdm
import json
if __name__ == "__main__":
    root_dir = "/media/mlfavorfit/sda/include_product_tag_image_mask/output"
    datas = get_datas_with_mask(root_dir)

    id_arr, colors_arr, weights_arr = load_templates_features()
    results = []
    for image_path, mask_path in tqdm(datas, total=len(datas), leave=True):
        try:
            image = np.array(Image.open(image_path).convert("RGB").resize((128,128)))
            mask = np.array(Image.open(mask_path).convert("RGB").resize((128,128)))
            
            colors_weights_pair = color_extraction(image, mask, n_cluster=4, epochs=5)
            colors, weights = run(colors_weights_pair)

            # Get indices
            similarity = calculate_cos_similarity(concat_array([colors.flatten()], weights*30, axis=1), concat_array(colors_arr, weights_arr*30, axis=1))
            indices = get_close_index(similarity, id_arr, max_num=30)

            temp = {"fn":image_path, "features":colors_weights_pair, "recommends":[colors.tolist(), weights.tolist()], "id":indices}
            results.append(temp)
        except:
            print("Error")
            continue
    with open("product_template_matching.json", mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent= 4)

        
        



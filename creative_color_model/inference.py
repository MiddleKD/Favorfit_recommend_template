import numpy as np
from .utils.colors_utils import extract_features_119
from .utils.data_utils import load_colors_540
from .models.inference import run
from .utils.data_utils import get_top_indices_and_probabilities
from .utils.colors_utils import color_filter_with_mask
from .utils.image_converter import ImageConverter

def inference(img_pil, mask_pil, resize_shape=[128,128]):
    image_np = np.array(img_pil.convert("RGB").resize(resize_shape))

    if mask_pil is not None:
        mask_np = np.array(mask_pil.convert("RGB").resize(resize_shape))
    else:
        mask_np = np.ones_like(image_np) * 255

    img_converter = ImageConverter(mode="rgb")

    masked_img_np = color_filter_with_mask(image_np, mask_np, 10)
    image_file = img_converter.np_to_image_without_saving(masked_img_np)


    list_of_colors = load_colors_540()
    input_data = extract_features_119(image_file, list_of_colors)

    probabilities_540 = run(input_data)

    colors, weights = get_top_indices_and_probabilities(
        probabilities_540, list_of_colors
    )

    return colors.tolist(), weights.tolist()[0]

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.getcwd(), ".."))
    
    from PIL import Image
    img_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos.jpg")
    mask_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos_mask.jpg")
    result = inference(img_pil=img_pil, mask_pil=None)
    print(result)

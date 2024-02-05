import numpy as np
from .utils.colors_utils import color_extraction
from .models.inference import run

def inference(img_pil, mask_pil, resize_shape=[128,128]):
    image_np = np.array(img_pil.convert("RGB").resize(resize_shape))

    if mask_pil is not None:
        mask_np = np.array(mask_pil.convert("RGB").resize(resize_shape))
    else:
        mask_np = np.ones_like(image_np) * 255

    colors_weights_pair = color_extraction(image_np, mask_np, n_cluster=4, epochs=5)
    colors, weights = run(colors_weights_pair)

    return colors.astype(np.uint8).tolist(), weights.tolist()

if __name__ == "__main__":
    from PIL import Image
    img_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos.jpg")
    mask_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos_mask.jpg")
    result = inference(img_pil=img_pil, mask_pil=None)
    print(result)

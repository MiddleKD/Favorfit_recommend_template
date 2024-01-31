import sys, os
sys.path.append(os.path.join(os.getcwd(), "similar_color_model"))
sys.path.append(os.path.join(os.getcwd(), "creative_color_model"))

from similar_color_model.inference import inference as inference_similar
from creative_color_model.inference import inference as inference_creative

def inference(img_pil, mask_pil):

    similar_colors, similar_weights = inference_similar(img_pil=img_pil, mask_pil=mask_pil)
    creative_colors, creative_weights = inference_creative(img_pil=img_pil, mask_pil=mask_pil)


    return {"similar_colors":similar_colors, "similar_weights":similar_weights,
            "creative_colors":creative_colors, "creative_weights":creative_weights}

if __name__ == "__main__":
    from PIL import Image

    img_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos.jpg")
    mask_pil = Image.open("/media/mlfavorfit/sda/include_product_tag_image_mask/output/0_aclos_mask.jpg")
    result = inference(img_pil=img_pil, mask_pil=None)
    print(result)

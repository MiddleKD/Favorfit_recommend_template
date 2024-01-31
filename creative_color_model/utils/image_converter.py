import base64
from io import BytesIO

import numpy as np
from PIL import Image


# 기본 input은 rgb로 가정합니다.
class ImageConverter:
    def __init__(self, mode="rgb"):
        self.mode = mode

    def pil_to_np(self, img_pil):
        img_np = np.array(img_pil)
        return img_np.astype(np.uint8)

    def b64_to_np(self, img_b64):
        img_pil = Image.open(BytesIO(base64.b64decode(img_b64.split(",")[1])))
        return self.pil_to_np(img_pil)

    def np_to_pil(self, img_np):
        img_pil = Image.fromarray(img_np)
        return img_pil

    def b64_to_pil(self, img_b64):
        return self.np_to_pil(self.b64_to_np(img_b64))

    def pil_to_b64(self, img_pil, type="jpg"):
        img_pil = img_pil.copy()
        img_data = BytesIO()
        img_pil.save(img_data, type)  # pick your format
        img_b64 = base64.b64encode(img_data.getvalue())
        img_b64 = f"data:img/{type};base64," + img_b64.decode("utf-8")
        return img_b64

    def np_to_b64(self, img_np):
        return self.pil_to_b64(self.np_to_pil(img_np))

    def convert(self, img, astype=None, channel=None, resize_shape=None):
        if isinstance(img, Image.Image):
            latent_img = img
            astype = "pil" if astype is None else astype
        elif isinstance(img, np.ndarray):
            latent_img = self.np_to_pil(img)
            astype = "np" if astype is None else astype
        elif isinstance(img, str) and "base64" in img:
            latent_img = self.b64_to_pil(img)
            astype = "b64" if astype is None else astype
        else:
            print("Unknown input image type: {}".format(type(img)))
            raise AssertionError

        if channel is not None:
            if channel == 2 or channel == "L":
                latent_img = latent_img.convert("L")
            elif channel == 3 or channel == "RGB":
                latent_img = latent_img.convert("RGB")
            elif channel == 4 or channel == "RGBA":
                latent_img = latent_img.convert("RGBA")
            else:
                print("Unknown image color type: {}".format(channel))
                raise AssertionError

        if resize_shape is not None:
            latent_img = latent_img.resize(resize_shape)

        if astype is not None:
            if astype == "pil":
                pass
            elif astype == "np":
                latent_img = self.pil_to_np(latent_img)
            elif astype == "b64":
                latent_img = self.pil_to_b64(latent_img)
            else:
                print("Unknown target image type: {}".format(astype))
                raise AssertionError

        return latent_img

    def base64_to_image(self, base64_string):
        data_start_index = base64_string.find(",") + 1
        base64_data = base64_string[data_start_index:]
        decoded_image = base64.b64decode(base64_data)

        image_file = BytesIO(decoded_image)

        return image_file

    def np_to_image_without_saving(self, img_np):
        img_pil = Image.fromarray(img_np)

        img_bytes_io = BytesIO()
        img_pil.save(img_bytes_io, format="PNG")
        img_bytes_io.seek(0)

        img_bytes = img_bytes_io.read()

        return BytesIO(img_bytes)

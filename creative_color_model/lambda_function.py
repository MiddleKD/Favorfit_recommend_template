import json


def respond(err, res):
    respond_msg = {
        "statusCode": 502 if err is not None else 200,
        "body": json.dumps(res),
    }
    print(f"Respond Message: {str(respond_msg)}")
    return respond_msg


def update_state(state):
    print(state)
    return state


def lambda_handler(event, context):
    state = update_state("Initializing")

    state = update_state("Importing external libraries")

    # load args
    state = update_state("Loading arguments")

    args = event["body"]
    if isinstance(args, str):
        args = json.loads(args)

    # load image
    state = update_state("Loading image")
    if "image_b64" in args:
        img = args["image_b64"]
    else:
        print("Can not find img in args")
        raise AssertionError

    from .utils.image_converter import ImageConverter
    img_converter = ImageConverter(mode="rgb")
    image_np = img_converter.convert(img, astype="np", channel=3, resize_shape=(128, 128))
    image_file = img_converter.base64_to_image(img)

    # load mask
    state = update_state("Loading mask")
    if "mask" in args:
        mask = args["mask"]
        mask_np = img_converter.convert(mask, astype="np", channel=3, resize_shape=(128, 128))
    else:
        print("Can not find mask in args")
        mask_np = None

    state = update_state("Get colors")
    from .utils.colors_utils import color_filter_with_mask
    masked_img_np = color_filter_with_mask(image_np, mask_np, 10)
    image_file = img_converter.np_to_image_without_saving(masked_img_np)

    # extract 119 features (input data)
    from .utils.colors_utils import extract_features_119
    from .utils.data_utils import load_colors_540
    list_of_colors = load_colors_540()
    input_data = extract_features_119(image_file, list_of_colors)

    from models.inference import run
    probabilities_540 = run(input_data)

    from .utils.data_utils import get_top_indices_and_probabilities
    colors, weights = get_top_indices_and_probabilities(
        probabilities_540, list_of_colors
    )

    # load background data
    state = update_state("Loading background data")
    from .utils.data_utils import load_templates_features
    templates = args["template_info"]

    id_arr, colors_arr, weights_arr = load_templates_features(templates)

    # get indices
    state = update_state("Get indices")
    from .utils.data_utils import calculate_cos_similarity, get_close_index, concat_array
    similarity = calculate_cos_similarity(
        concat_array([colors.flatten()], weights * 30, axis=1),
        concat_array(colors_arr, weights_arr * 30, axis=1),
    )
    indices = get_close_index(similarity, id_arr)

    # convert RGB to hex
    state = update_state("Converting RGB to hex")
    from .utils.colors_utils import colors_to_hex
    colors_hex = colors_to_hex(colors)

    # Respond
    state = update_state("Responding")
    output = {"colors": colors_hex, "id": indices}

    return respond(None, output)


if __name__ == "__main__":
    lambda_handler({}, None)

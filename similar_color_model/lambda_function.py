import json

def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    print('Respond Message: {}'.format(str(respond_msg)))
    return respond_msg


def update_state(state):
    print(state)
    return state


def lambda_handler(event, context):
    state = update_state('Initializing')

    # Import
    state = update_state('Importing external libraries')


    # Load args
    state = update_state('Loading arguments')
    print('Event: {}'.format(event))

    args = event["body"]
    if isinstance(args, str):
        args = json.loads(args)

    print('Arguments: {}'.format(str(args)))


    # Load image
    state = update_state('Loading image')
    if 'image_b64' in args:
        img = args['image_b64']
    else:
        print('Can not find img in args')
        raise AssertionError
    
    from utils.image_converter import ImageConverter
    img_converter = ImageConverter(mode="rgb")
    img_np = img_converter.convert(img, astype="np", channel=3, resize_shape=(128, 128))


    # Load Mask
    state = update_state('Loading mask')
    if 'mask' in args:
        mask = args['mask']
        mask_np = img_converter.convert(mask, astype="np", channel=3, resize_shape=(128, 128))
    else:
        print('Can not find mask in args')
        mask_np = None

    # Get colors
    from utils.colors_utils import color_extraction
    state = update_state('Get colors')
    colors_weights_pair = color_extraction(img_np, mask_np, n_cluster=4, epochs=5)

    from models.inference import run
    colors, weights = run(colors_weights_pair)


    # Load background data
    from utils.data_utils import load_templates_features
    state = update_state('Loading background data')
    id_arr, colors_arr, weights_arr = load_templates_features()


    # Get indices
    from utils.data_utils import calculate_cos_similarity, get_close_index, concat_array
    state = update_state('Get indices')
    similarity = calculate_cos_similarity(concat_array([colors.flatten()], weights*30, axis=1), concat_array(colors_arr, weights_arr*30, axis=1))
    indices = get_close_index(similarity, id_arr)


    # Convert RGB to hex
    from utils.colors_utils import colors_to_hex
    state = update_state('Converting RGB to hex')
    colors_hex = colors_to_hex(colors[0])


    # Respond
    state = update_state('Responding')
    output = {"colors": colors_hex, 'id': indices}

    return respond(None, output)

if __name__ == "__main__":
    lambda_handler({}, None)
import numpy as np
import json
from ..models.model import forward
import os


def run(input_data):
    relative_path = '20240129.json'
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

    with open(file_path) as rg:
        model_weights = json.load(rg)

    for key in model_weights:
        model_weights[key] = np.array(model_weights[key])

    # 모델 연산 수행
    output = forward(input_data, model_weights)
    return output


if __name__ == "__main__":
    run()

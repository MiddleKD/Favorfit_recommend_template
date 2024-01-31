import numpy as np

def layer_norm(input, gamma, beta, eps=1e-5):
    mean = input.mean(axis=-1, keepdims=True)
    std = input.std(axis=-1, keepdims=True)
    return gamma * (input - mean) / (std + eps) + beta

def relu(input):
    return np.maximum(input, 0)

def linear(input, weight, bias):
    return np.matmul(input, weight.T) + bias

def residual_block(input, block_weights, block_biases, block_weights_norm, block_biases_norm):
    output = input
    output = linear(output, block_weights, block_biases)
    output = layer_norm(output, gamma=block_weights_norm, beta=block_biases_norm)
    output = relu(output)
    return output + input


def forward(input, model_weights):
    # 첫 번째 선형 레이어
    weight, bias = model_weights['first_layer.weight'], model_weights['first_layer.bias']
    output = linear(input, weight, bias)

    expand_dims = [32, 64, 128]
    inter_num = 3
    # 확장 레이어
    for i in range(len(expand_dims) - 1):
        weight, bias = model_weights[f'expander.{i}.0.weight'], model_weights[f'expander.{i}.0.bias']
        weight_norm, bias_norm = model_weights[f'expander.{i}.1.weight'], model_weights[f'expander.{i}.1.bias']
        output = linear(output, weight, bias)
        output = layer_norm(output, gamma=weight_norm, beta=bias_norm)
        output = relu(output)

    # 중간 레이어
    for i in range(inter_num):
        block_weights = model_weights[f'intermediate.{i}.0.block.0.weight']
        block_biases = model_weights[f'intermediate.{i}.0.block.0.bias']
        block_weights_norm = model_weights[f'intermediate.{i}.0.block.1.weight']
        block_biases_norm = model_weights[f'intermediate.{i}.0.block.1.bias']
        output = residual_block(output, block_weights, block_biases, block_weights_norm, block_biases_norm)

    # 마지막 레이어
    weight, bias = model_weights['last_layer.0.weight'], model_weights['last_layer.0.bias']
    output = linear(output, weight, bias)

    return output


import json
def make_weight_json(model, save_path):

    model_np = {k: v.numpy().tolist() for k, v in model.state_dict().items()}

    # JSON 파일로 저장
    with open(save_path, 'w') as f:
        json.dump(model_np, f)
    

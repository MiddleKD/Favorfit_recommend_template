import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


def linear(x, weight, bias):
    return np.matmul(x, weight.T) + bias


def relu(x):
    return np.maximum(0, x)


def forward(x, model_weights):
    # Linear 1
    x = linear(
        x, model_weights["seq_modules.0.weight"], model_weights["seq_modules.0.bias"]
    )
    # LayerNorm 1
    x = layer_norm(
        x,
        gamma=model_weights["seq_modules.1.weight"],
        beta=model_weights["seq_modules.1.bias"],
    )
    x = relu(x)

    # Linear 2
    x = linear(
        x, model_weights["seq_modules.3.weight"], model_weights["seq_modules.3.bias"]
    )
    # LayerNorm 2
    x = layer_norm(
        x,
        gamma=model_weights["seq_modules.4.weight"],
        beta=model_weights["seq_modules.4.bias"],
    )
    x = relu(x)

    # Output Layer
    x = linear(x, model_weights["out.weight"], model_weights["out.bias"])

    return x

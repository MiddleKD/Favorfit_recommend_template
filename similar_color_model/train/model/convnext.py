from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
import torch.nn as nn


class ConvnextDecoder(nn.Module):
    def __init__(self, out_channels=16, decoder_dims=[512, 128, 32]):
        super().__init__()
        
        self.connext_layer = convnext_tiny()

        modules = []
        in_d_dim = 1000
        for d_dim in decoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_d_dim, d_dim),
                    nn.LayerNorm(d_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.3)
                )
            )
            in_d_dim = d_dim
        self.decoder_block = nn.Sequential(*modules)

        # Final fully connected layers
        self.fc = nn.Linear(decoder_dims[-1], out_channels)

    def forward(self, x):
        out = self.connext_layer(x)
        # out = self.decoder_block(out)
        out = self.fc(out)
        return out

# Example usage:
# model = CNNEncoder(in_channels=3, out_channels=10, hidden_dims=[64, 128, 256, 512])

if __name__ == "__main__":
    img = torch.randn([1, 3, 224,224])
    model = ConvnextDecoder(out_channels=16, decoder_dims=[512, 128, 32])

    model.eval()
    print(model(img).shape)
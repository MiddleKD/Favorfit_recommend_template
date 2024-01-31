import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, hidden_dims=[32, 64, 128, 256, 512]):
        super(CNNEncoder, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1)
        self.initial_bn = nn.BatchNorm2d(hidden_dims[0])
        self.initial_relu = nn.ReLU(inplace=True)

        # Build the Residual Blocks
        layers = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(ResidualBlock(in_channels, h_dim, stride=2,
                            downsample=nn.Sequential(
                                nn.Conv2d(in_channels, h_dim, kernel_size=1, stride=2, bias=False),
                                nn.BatchNorm2d(h_dim)
                            )))
            in_channels = h_dim

        self.residual_blocks = nn.Sequential(*layers)

        # Final fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.initial_relu(out)
        out = self.residual_blocks(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)
        return out

# Example usage:
# model = CNNEncoder(in_channels=3, out_channels=10, hidden_dims=[64, 128, 256, 512])

if __name__ == "__main__":
    # Define your network
    hidden_dims = [32, 64, 128, 256, 512]
    model = CNNEncoder(in_channels=3, hidden_dims=hidden_dims, out_channels=10)

    # Example input
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(output)
    print(output.shape)  # Should be torch.Size([1, 10])

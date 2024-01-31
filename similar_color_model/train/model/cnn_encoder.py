import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, hidden_dims=[32, 64, 128, 256, 512], num_inter_layers = 3):
        super().__init__()


        # encoder
        modules = []
        en_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(en_in_channels, h_dim, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                ))
            en_in_channels = h_dim
        modules.append(nn.Sequential(nn.MaxPool2d(2,2)))   
        self.encoder = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # intermediate
        modules = [nn.Sequential(nn.Flatten())]
        inter_in_channels = hidden_dims[-1]
        for _ in range(num_inter_layers):
            modules.append(
                nn.Sequential(
                    nn.Linear(inter_in_channels, inter_in_channels),
                    nn.LayerNorm(inter_in_channels),
                    nn.LeakyReLU(),
                ))
        self.intermediate = nn.Sequential(*modules)


        # decoder
        modules = []
        de_in_channels = hidden_dims[-1]
        for h_dim in hidden_dims[::-1]:
            modules.append(
                nn.Sequential(
                    nn.Linear(de_in_channels, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                ))
            de_in_channels = h_dim
        self.decoder = nn.Sequential(*modules)


        # fully connected
        self.fc = nn.Linear(hidden_dims[0], out_channels)


    def forward(self, img):
        encoded = self.encoder(img)
        encoded = self.adaptive_pool(encoded)
        intermediate = self.intermediate(encoded)
        decoded = self.decoder(intermediate)
        result = self.fc(decoded)
        return result
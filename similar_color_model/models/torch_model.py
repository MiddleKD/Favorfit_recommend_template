import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x) + x
    

class LinearRes(nn.Module):
    def __init__(self, input_size, output_size, inter_num=3, expand_dims=[32, 64, 128]):
        super().__init__()

        self.first_layer = nn.Linear(input_size, expand_dims[0])


        module = []
        in_size = expand_dims[0]
        for ex_dim in expand_dims[1:]:
            module.append(nn.Sequential(
                                nn.Linear(in_size, ex_dim),
                                nn.LayerNorm(ex_dim),
                                nn.ReLU(),
                            ))
            in_size = ex_dim
        self.expander = nn.Sequential(*module)


        module = []
        for _ in range(inter_num):
            module.append(nn.Sequential(
                                ResidualBlock(expand_dims[-1]),
                            ))
        self.intermediate = nn.Sequential(*module)

        self.last_layer = nn.Sequential(
                                nn.Linear(expand_dims[-1], output_size),
                                )


    def forward(self, x):
        x = self.first_layer(x)
        x = self.expander(x)
        x = self.intermediate(x)
        x = self.last_layer(x)
        return x

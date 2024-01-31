import torch
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
    

class ReversedAutoEncoder(nn.Module):
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
                                # nn.Dropout(0.3),
                            ))
        self.intermediate = nn.Sequential(*module)
        self.dropout = nn.Dropout(0.3)

        module = []
        in_size = expand_dims[-1]
        for re_dim in reversed(expand_dims[:-1]):
            module.append(nn.Sequential(
                                nn.Linear(in_size, re_dim),
                                nn.LayerNorm(re_dim),
                                nn.ReLU(),
                            ))
            in_size = re_dim
        self.reducer = nn.Sequential(*module)


        self.last_layer = nn.Sequential(
                                nn.Linear(expand_dims[0], output_size),
                                )


    def forward(self, x):
        x = self.first_layer(x)
        x = self.expander(x)
        # x = self.dropout(x)
        x = self.intermediate(x)
        x = self.reducer(x)
        x = self.last_layer(x)
        return x


class ReversedVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_dim, inter_num=3, expand_dims=[32,64]):
        super().__init__()

        # 첫 번째 계층
        self.first_layer = nn.Linear(input_size, expand_dims[0])

        # 확장 계층
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

        # 중간 계층 (intermediate)
        module = []
        for _ in range(inter_num):
            module.append(nn.Sequential(
                ResidualBlock(expand_dims[-1]),
                # nn.Dropout(0.5),
            ))
        self.intermediate = nn.Sequential(*module)

        # 인코더의 평균과 로그 분산 계층
        self.fc_mu = nn.Linear(expand_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(expand_dims[-1], latent_dim)

        # 디코더의 첫 번째 계층
        self.fc_decode = nn.Linear(latent_dim, expand_dims[-1])

        # 축소 계층
        module = []
        in_size = expand_dims[-1]
        for re_dim in reversed(expand_dims[:-1]):
            module.append(nn.Sequential(
                nn.Linear(in_size, re_dim),
                nn.LayerNorm(re_dim),
                nn.ReLU(),
            ))
            in_size = re_dim
        self.reducer = nn.Sequential(*module)

        # 마지막 계층
        self.last_layer = nn.Sequential(
            nn.Linear(expand_dims[0], output_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.first_layer(x)
        x = self.expander(x)
        # x = self.intermediate(x)

        # 인코더를 통한 잠재 공간의 평균과 로그 분산 계산
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # 리파라미터화 트릭 적용
        z = self.reparameterize(mu, logvar)

        # 디코더를 통한 데이터 재구성
        x = self.fc_decode(z)
        x = self.reducer(x)
        x = self.last_layer(x)
        return x, mu, logvar


if __name__ == "__main__":
    import torch

    input = torch.randn([8,16])
    print(input)

    model = ReversedAutoEncoder(input_size=16, output_size=16, inter_num=3)
    output = model(input)

    print(output)

    model = ReversedVAE(input_size=16, output_size=16, latent_dim=128, inter_num=3)
    output = model(input)
    print(output)
import torch
import torch.nn as nn
import torch.nn.functional as F

class VarianceSensitiveHuberLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', delta: float = 1.0, gamma: float = 0.5) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta
        self.gamma = gamma

    def forward(self, input, target):
        i_colors = input.reshape(-1,4,4)[:,:,:3].reshape(-1,12)
        hue__val_i_colors = input.reshape(-1,4,4)[:,:,[0,2]].reshape(-1,8)
        hue__val_t_colors = target.reshape(-1,4,4)[:,:,[0,2]].reshape(-1,8)

        h_loss = F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)
        hv_loss = F.huber_loss(hue__val_i_colors, hue__val_t_colors, reduction=self.reduction, delta=self.delta)

        variance_color = torch.var(i_colors, dim=1).mean()
        variance_batch = torch.var(i_colors, dim=0).mean()
        
        vc_loss = self.gamma *(3/torch.exp(0.5*variance_color)+0.00001)
        vb_loss = self.gamma *(3/torch.exp(0.5*variance_batch)+0.00001)

        return h_loss + vc_loss + vb_loss + hv_loss

class VarianceSensitiveMSELoss(nn.Module):
    def __init__(self, gamma: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        i_colors = input.reshape(-1,4,4)[:,:,:3].reshape(-1,12)
        # hue__val_i_colors = input.reshape(-1,4,4)[:,:,[0,2]].reshape(-1,8)
        # hue__val_t_colors = target.reshape(-1,4,4)[:,:,[0,2]].reshape(-1,8)

        mse_loss = F.mse_loss(input, target)
        # hv_loss = F.mse_loss(hue__val_i_colors, hue__val_t_colors)

        variance_color = torch.var(i_colors, dim=1).mean()
        variance_batch = torch.var(i_colors, dim=0).mean()
        
        # vc_loss = self.gamma *(3/torch.exp(0.5*variance_color)+0.00001)
        # vb_loss = self.gamma *(3/torch.exp(0.5*variance_batch)+0.00001)
        vc_loss = self.gamma *(20/torch.exp(5*variance_color)+0.00001)
        vb_loss = self.gamma *(20/torch.exp(5*variance_batch)+0.00001)
        # vc_loss = -self.gamma*variance_color + self.gamma/2
        # vb_loss = -self.gamma*variance_batch + self.gamma/2

        return mse_loss + vc_loss + vb_loss # + hv_loss


class MSEKLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss
    

if __name__ == "__main__":
    a = torch.tensor([[-2.0494049148043496, 0.8704481792717087, 1.751111111111111,0.4], [1.6666666666666667, -1.9481792717086837, -1.612723311546841,0.3], [1.8721637126466306, -1.4929971988795518, -1.4035729847494554,0.2], [0.8789279904101379, 0.45028011204481794, 0.9493681917211328,0.1]],
                     dtype=torch.float32).reshape(1,-1)
    b = torch.tensor([[10,10,10,11,11,11,10,10, 1,1,1,11,11,103,103,102]], dtype=torch.float32)

    criterion = VarianceSensitiveHuberLoss(reduction="mean", delta=0.6)
    print(criterion(a, b))
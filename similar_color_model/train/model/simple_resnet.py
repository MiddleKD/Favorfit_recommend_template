
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.bn(out)
        out += residual
        out = self.relu(out)
        return out

class SimpleNNWithResBlocks(nn.Module):
    def __init__(self):
        super(SimpleNNWithResBlocks, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.res1 = ResidualBlock(32)
        self.fc2 = nn.Linear(32, 64)
        self.res2 = ResidualBlock(64)
        self.fc3 = nn.Linear(64, 128)
        self.res3 = ResidualBlock(128)
        self.fc5 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        x = self.res2(x)
        x = F.relu(self.fc3(x))
        x = self.res3(x)

        x = self.fc5(x)
        
        return x

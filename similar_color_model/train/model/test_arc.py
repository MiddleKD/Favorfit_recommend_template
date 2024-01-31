import torch
import torch.nn as nn

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

class TestArc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.last = nn.Linear(32, 16)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.last(x)
        
        return x


# 모델 정의
class SeqRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=4):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN 층 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 출력 층 정의
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        print(x.shape)
        x = x.reshape(len(x), 4, 4)
        print(x.shape)
        # RNN 순방향 전파
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        print(out.shape)
        print(_.shape)
        out = self.fc(out)
        return out.reshape(len(out), 16)


# RNN Cell 클래스 정의
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        return hidden

# RNN 네트워크 클래스 정의
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)

    def forward(self, input):
        hidden = torch.zeros(input.size(0), self.hidden_size)
        hidden_states = []

        for i in range(input.size(1)):
            hidden = self.rnn_cell(input[:, i], hidden)
            hidden_states.append(hidden)

        return torch.stack(hidden_states, dim=1)



if __name__ == "__main__":
    # 모델 초기화 및 입력 데이터 준비
    input_size = 5
    hidden_size = 10
    seq_len = 6
    batch_size = 1

    model = RNN(input_size, hidden_size)
    input = torch.randn(batch_size, seq_len, input_size)

    # 모델 실행 및 히든 상태 확인
    with torch.no_grad():
        hidden_states = model(input)
    
    # 히든 상태의 크기 확인 (배치 크기, 시퀀스 길이, 히든 크기)
    print(hidden_states.shape)

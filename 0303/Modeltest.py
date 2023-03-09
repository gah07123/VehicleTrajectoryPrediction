import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 定义LSTM网络
class Mylstm(nn.Module):
    """
    seq:sequence length
        input_size:feature size
        hidden_size:hidden layer size
        output_size:output size(after linear layer)
        h0:initial h
        c0:initial c
        ht:every layer last h
        out:h of every time and last layer in the sequence
    """
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1, batch_first=False, dropout=0,
                 dropout_fc: float = 0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, input):
        # input:[seq, batch_size, input_size]
        output, (ht, ct) = self.lstm(input)
        # output: [seq, batch_size, hidden_size]
        seq, bs, h = output.shape
        # 取出最后一个时刻的预测值hidden_t
        hidden_t = torch.cat([ht[-1]], dim=1)
        # Dropout
        hidden_t = self.dropout_fc(hidden_t)
        # 过全连接层
        out = self.fc(hidden_t)

        return out, (ht, ct)

def readtxt(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split()
            data.append(line)
    data = np.array(data, dtype="float32")
    # print(data.shape[0])
    return data

def Normalize(data):
    data_mean = torch.mean(data)
    data_std = torch.std(data)
    data_norm = (data - data_mean) / data_std
    return data_norm

# 定义数据集
class MyDataset(Dataset):
    def __init__(self, input, output):
        self.data_input = torch.from_numpy(input)
        self.data_output = torch.from_numpy(output)
        self.len = self.data_output.shape[0]

    def __getitem__(self, idx):
        return self.data_input[idx], self.data_output[idx]

    def __len__(self):
        return self.len

# 原始数据
train_input = np.array(readtxt("training_data_input.txt"))
test_input = np.array(readtxt("testing_data_input.txt"))
data_input = np.concatenate((train_input, test_input), axis=0)
data_input = data_input.reshape(data_input.shape[0] * data_input.shape[1], -1)

input_seq = np.zeros(0, dtype='float32')
output_seq = np.zeros(0, dtype='float32')
# 滑动窗口处理
for i in range(data_input.shape[0] - 10):
    input_seq = np.append(input_seq, data_input[i:i + 9])
    output_seq = np.append(output_seq, data_input[i + 10])
input_seq = input_seq.reshape(-1, 9)
output_seq = output_seq.reshape(-1, 1)

train_ratio = 0.5
train_len = int(train_ratio * input_seq.shape[0])
train_input_seq = input_seq[:train_len]
train_output_seq = output_seq[:train_len]
test_input_seq = input_seq[train_len:]
test_output_seq = output_seq[train_len:]

# 数据集导入
train_dataset = MyDataset(train_input_seq, train_output_seq)
test_dataset = MyDataset(test_input_seq, test_output_seq)
# 标准化
data_input_norm = torch.cat([train_dataset.data_input, test_dataset.data_input], dim=0)
data_output_norm = torch.cat([train_dataset.data_output, test_dataset.data_output], dim=0)
data_input_norm = Normalize(data_input_norm)
data_output_norm = Normalize(data_output_norm)
train_dataset.data_input = data_input_norm[:train_len]
train_dataset.data_output = data_output_norm[:train_len]
test_dataset.data_input = data_input_norm[train_len:]
test_dataset.data_output = data_output_norm[train_len:]

# 加载模型
model = torch.load("../GraphsandLogs/model0227do0.2wd0.001ld0.5.pth")

# 预测，切换回CPU
model = model.eval().cpu()
real_output = torch.cat([train_dataset.data_output, test_dataset.data_output], dim=0)
real_output = real_output.numpy()

pred_input = torch.cat([train_dataset.data_input, test_dataset.data_input], dim=0)
predictions = np.zeros(0)

for i in range(pred_input.shape[0]):
    with torch.no_grad():
        pred_train_input = pred_input[i].view(9, 1, 1)
        pred, _ = model(pred_train_input)
        predictions = np.append(predictions, pred.numpy())

time_steps = np.linspace(1, pred_input.shape[0], pred_input.shape[0], dtype=int)
time_steps = time_steps.reshape(time_steps.shape[0], 1)
predictions = predictions.reshape(predictions.shape[0], 1)
difference = np.abs(predictions - real_output)
# difference = np.divide(difference, real_output)
# 必须降维，否则会报size-1 xxx错误
difference = np.squeeze(difference)
# 取误差最大值
difference_max_idx = np.argmax(difference)
difference_max = np.max(difference)


plt.xlabel('time_steps')
plt.ylabel('Y(Normalized)')
plt.plot(real_output, "r", label="real_pos")
plt.plot(predictions, "g--", label="pred_pos")
plt.plot((train_len, train_len), (-1.5, 1.5), "b--")
plt.plot(difference, "aqua", label="error")
plt.annotate('max=%.3f' % difference_max, xy=(difference_max_idx - 5, difference_max + 0.05))
plt.legend(loc="best", fontsize=10)
plt.savefig('../GraphsandLogs/model0227do0.2wd0.001ld0.5.png', format='png', dpi=200)
plt.show()


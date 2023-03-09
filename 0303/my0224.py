# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import time

# 9个点预测一个点
# 主要问题在于维度匹配，这是一个many-to-one的问题，需要取最后一个时刻的ht而不是整个output
# 如果输入了h0，c0，还要使用GPU训练，要确保输入之后h0和c0位于GPU中
# -2023.02.24- 使用了滑动窗口来作为输入
# bug：修正了train_dataset的input
# -2023.02.27- dropout_fc -> 0.2 weight_decay -> 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputsize = 1
batchsize = 512
outputsize = 1
seq_length = 9

do_fc = 0.2  # 0.8 -> 0.2 -> 0.4 -> 0.2
w_decay = 0  # 0.001 -> 0
lrate_decay_step = 600
lrate_decay_gamma = 1

h_size = 128  # 128->144->128->64->150->128
layers = 1
lrate = 0.001
bidir = False
epoch = 2000
print_time = False

# 读取txt文件
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
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1, batch_first=False,
                 dropout: float = 0, dropout_fc: float = 0):
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

def Normalize(data):
    data_mean = torch.mean(data)
    data_std = torch.std(data)
    data_norm = (data - data_mean) / data_std
    return data_norm


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

# Dataloader初始化
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False,
                          pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False,
                         pin_memory=True)
# 模型初始化
model = Mylstm(input_size=inputsize, hidden_size=h_size, output_size=outputsize, num_layers=layers,
               dropout_fc=do_fc, dropout=0)
model = model.to(device)

criterion = nn.MSELoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=w_decay)
# 学习率衰减
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lrate_decay_step,
                                            gamma=lrate_decay_gamma, verbose=False)
losses = []

# if bidir == False:
#     h0 = torch.zeros([1 * layers, batchsize, h_size])
#     c0 = torch.zeros([1 * layers, batchsize, h_size])
# else:
#     h0 = torch.zeros([2 * layers, batchsize, h_size])
#     c0 = torch.zeros([2 * layers, batchsize, h_size])

writer = SummaryWriter("logs")
model = model.train()
t = 0

for iter in range(epoch):
    loss = torch.zeros(0)
    start_time = time.perf_counter()
    # 遍历数据集
    for data in train_loader:

        data_in, data_out = data
        # input -> [seq, batch_size, input_size]，匹配输入维度
        data_in = data_in.view(data_in.shape[1], data_in.shape[0], inputsize)
        # 存入GPU
        data_in = data_in.to(device)
        data_out = data_out.to(device)
        # 这里的output是最后一个时刻的ht
        output, (ht, ct) = model(data_in)
        # 反向传播
        loss = criterion(output, data_out)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    schedular.step()
    end_time = time.perf_counter()
    t = t + 1
    if print_time:
        print(str(t) + "  " + str(end_time - start_time))

    losses.append(loss.item())
    if iter % 50 == 0:
        print("Iteration: {}  Loss: {:.6f}".format(iter, loss.item()))
        # 记录loss曲线
        writer.add_scalar("loss" + str(lrate_decay_gamma) + str(w_decay) + str(h_size), loss.item(), iter)
writer.close()

# 保存模型
torch.save(model, "model.pth")

# 预测，切换回CPU
model = model.eval().cpu()
real_output = torch.cat([train_dataset.data_output, test_dataset.data_output], dim=0)

pred_input = torch.cat([train_dataset.data_input, test_dataset.data_input], dim=0)
predictions = np.zeros(0)
# losses_test = []

for i in range(pred_input.shape[0]):
    with torch.no_grad():
        pred_train_input = pred_input[i].view(9, 1, 1)
        pred, _ = model(pred_train_input)
        predictions = np.append(predictions, pred.numpy())


plt.subplots(1, 1)
plt.plot(losses, "r")
plt.savefig('loss-ld' + str(lrate_decay_gamma) + '.png', format='png', dpi=200)

plt.subplots(1, 1)
plt.plot(real_output.numpy(), "r")
plt.plot(predictions, "g--")
plt.plot((train_len, train_len), (-1, 1), "b--")
plt.savefig('test.png', format='png', dpi=200)
plt.show()












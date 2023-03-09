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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputsize = 1
batchsize = 256
outputsize = 1
seq_length = 9

h_size = 16
layers = 1
lrate = 0.001
bidir = False
epoch = 1000

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
    def __init__(self, dir_input, dir_output):
        self.data_input = readtxt(dir_input)
        self.data_output = readtxt(dir_output)
        self.len = self.data_output.shape[0]
        self.data_input = torch.from_numpy(self.data_input)
        self.data_output = torch.from_numpy(self.data_output)

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
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1, batch_first=False, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.layernorm = nn.LayerNorm([inputsize])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input:[seq, batch_size, input_size]
        # h_0 = h0.to(device)
        # c_0 = c0.to(device)
        output, (ht, ct) = self.lstm(input)
        # output: [seq, batch_size, hidden_size]
        seq, bs, h = output.shape
        # 取出最后一个时刻的预测值hidden_t
        hidden_t = torch.cat([ht[-1]], dim=1)
        # 过全连接层
        out = self.fc(hidden_t)
        # 标准化
        # out = self.layernorm(out)

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
# 压平
# data_input = data_input.reshape(data_input.shape[0] * data_input.shape[1], 1)

# plt.plot(data_input)
# plt.show()
# 数据集导入
train_dataset = MyDataset("training_data_input.txt", "training_data_output.txt")
test_dataset = MyDataset("testing_data_input.txt", "testing_data_output.txt")
# 标准化
data_input_norm = torch.cat([train_dataset.data_input, train_dataset.data_input], dim=0)
data_output_norm = torch.cat([train_dataset.data_output, test_dataset.data_output], dim=0)
data_input_norm = Normalize(data_input_norm)
data_output_norm = Normalize(data_output_norm)
train_dataset.data_input = data_input_norm[:1800]
test_dataset.data_input = data_input_norm[1800:]
train_dataset.data_output = data_output_norm[:1800]
test_dataset.data_output = data_output_norm[1800:]

# Dataloader初始化
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False,
                          pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False,
                         pin_memory=True)
# 模型初始化
model = Mylstm(input_size=inputsize, hidden_size=h_size, output_size=outputsize, num_layers=layers)
model = model.to(device)

criterion = nn.MSELoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
# 指数学习率衰减
explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = []

# if bidir == False:
#     h0 = torch.zeros([1 * layers, batchsize, h_size])
#     c0 = torch.zeros([1 * layers, batchsize, h_size])
# else:
#     h0 = torch.zeros([2 * layers, batchsize, h_size])
#     c0 = torch.zeros([2 * layers, batchsize, h_size])
writer = SummaryWriter("logs")
model = model.train()
for iter in range(epoch):
    # start_time = time.perf_counter()
    loss = torch.zeros(0)
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
        # explr.step()

    # end_time = time.perf_counter()
    # print(str(end_time - start_time))
    losses.append(loss.item())
    if iter % 50 == 0:
        print("Iteration: {}  Loss: {:.6f}".format(iter, loss.item()))
        writer.add_scalar("loss", loss.item(), iter)
writer.close()

# 保存模型
torch.save(model.state_dict(), "model0224.pth")

# 预测，切换回CPU
model = model.eval().cpu()
real_output = torch.cat([train_dataset.data_output, test_dataset.data_output], dim=0)

pred_input = torch.cat([train_dataset.data_input, test_dataset.data_input], dim=0)
predictions = np.zeros(0)

for i in range(pred_input.shape[0]):
    with torch.no_grad():
        pred_train_input = pred_input[i].view(9, 1, 1)
        pred, _ = model(pred_train_input)
        predictions = np.append(predictions, pred.numpy())

plt.subplots(1, 1)
plt.plot(losses, "r")
plt.subplots(1, 1)

plt.plot(real_output.numpy(), "r")
plt.plot(predictions, "g--")
plt.plot((1800, 1800), (-1, 1), "b--")
plt.show()












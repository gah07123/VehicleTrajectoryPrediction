import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch.mycode.Model import *
sys.path.append("..")
from pytorch.data_process.dataprocess import *

# 指定随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# 初始化权重参数，权重服从 N(0, 0.01) 分布，偏置初始化为0
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

# 训练模式
def train(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    # outputs = torch.zeros((1, ))

    for data in dataloader:
        src, trg = data
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        batch_size, trg_len, output_size = output.shape
        # trg = [batch size, trg len, output size]
        # output = [batch size, trg len, output size]

        output = output[:].view(-1, output_size)
        trg = trg[:].view(-1, output_size)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# 测试模式
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for data in dataloader:
            src, trg = data
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing
            output_size = output.shape[-1]

            output = output[:].view(-1, output_size)
            trg = trg[:].view(-1, output_size)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# 计算迭代时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    elapsed_ms = int(elapsed_time * 1000)
    return elapsed_mins, elapsed_secs, elapsed_ms

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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # 网络参数
    best_valid_loss = float('inf')
    src_length = 5
    trg_length = 5
    input_size = 4
    output_size = 4
    num_layers = 2
    enc_hid_size = 128
    dec_hid_size = 128
    # 训练参数
    batch_size = 16
    dropout = 0.5
    clip = 1
    max_epochs = 200
    learning_rate = 0.001
    train_ratio = 0.8

    # 加载数据
    data_car_group, data_car_id = data_process('../data_process/10_tracks.csv', '../data_process/10_tracksMeta.csv')
    train_len = int(train_ratio * len(data_car_group))
    valid_len = len(data_car_group) - train_len
    # 初始化各层网络
    attn = Seq2SeqAttentionMechanism(enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size)
    enc = Seq2SeqEncoder(input_size=input_size, enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size,
                         num_layers=num_layers, dropout=dropout)
    dec = Seq2SeqDecoder(input_size=input_size, enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size,
                         output_size=output_size, attention=attn, num_layers=num_layers, dropout=dropout)
    # 初始化模型
    model = Model(enc, dec, device).to(device)
    model.apply(init_weights)
    output = torch.zeros(batch_size, trg_length, output_size).to(device)
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # 迭代
    for epoch in range(max_epochs):
        start_time = time.perf_counter()
        train_losses = []
        valid_losses = []
        train_sequence = np.arange(train_len)
        valid_sequence = np.random.randint(0, valid_len, size=train_len)
        # 打乱顺序，使每个epoch抽取序列的顺序不同
        np.random.shuffle(train_sequence)

        for i, j in zip(train_sequence, valid_sequence):
            # 读取训练集和测试集
            data_group_input, data_group_output = slide_window(data_car_group[i], step_in=5, step_out=5, slide_step=2)
            data_valid_input, data_valid_output = slide_window(data_car_group[j], step_in=5, step_out=5, slide_step=2)
            dataset_train = MyDataset(data_group_input, data_group_output)
            dataset_valid = MyDataset(data_valid_input, data_valid_output)
            dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)
            dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)
            train_loss = train(model, dataloader_train, optimizer, criterion, clip, device)
            valid_loss = evaluate(model, dataloader_valid, criterion, device)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        end_time = time.perf_counter()

        epoch_mins, epoch_secs, epoch_ms = epoch_time(start_time, end_time)

        # 存疑，通过验证集来保存模型
        if max(valid_losses) < best_valid_loss:
            best_valid_loss = max(valid_losses)
            torch.save(model.state_dict(), 'model/test_out5.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch time(ms): {epoch_ms} ms')
        print(f'\tTrain Loss: {max(train_losses):.3f}')
        print(f'\tValid Loss: {max(valid_losses):.3f}')
        print(f'\tBest Valid Loss: {best_valid_loss:.3f}')



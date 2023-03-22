import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from pytorch.mycode.Model import *
from pytorch.data_process.dataprocess import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# 指定随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

# 测试模式
def test(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    batch_size, trg_len, output_size = dataloader.dataset.data_output.shape
    outputs = torch.zeros([0, trg_len, output_size]).to(device)

    with torch.no_grad():
        for data in dataloader:
            src, trg = data
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing
            # output_size = output.shape[-1]
            outputs = torch.cat((outputs, output), dim=0)

            output = output[:].view(-1, output_size)
            trg = trg[:].view(-1, output_size)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader), outputs

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    input_size = 4
    output_size = 4
    enc_hid_size = 128
    dec_hid_size = 128

    batch_size = 16
    num_layers = 2
    dropout = 0.5

    attn = Seq2SeqAttentionMechanism(enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size)
    enc = Seq2SeqEncoder(input_size=input_size, enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size,
                         num_layers=num_layers, dropout=dropout)
    dec = Seq2SeqDecoder(input_size=input_size, enc_hid_size=enc_hid_size, dec_hid_size=dec_hid_size,
                         output_size=output_size, attention=attn, num_layers=num_layers, dropout=dropout)

    model = Model(enc, dec, device).to(device)
    model.load_state_dict(torch.load("./model/test_out5.pt"))
    model.eval()
    # 加载数据
    seq_number = 4
    data_car_group, data_car_id = data_process('../data_process/10_tracks.csv', '../data_process/10_tracksMeta.csv')
    data_group_input, data_group_output = slide_window(data_car_group[seq_number], step_in=5, step_out=3, slide_step=3)
    print(data_car_id[seq_number])
    dataset_train = MyDataset(data_group_input, data_group_output)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
                                  drop_last=False)
    criterion = nn.MSELoss()

    loss, outputs = test(model, dataloader_train, criterion, device)
    outputs = outputs.view(-1, 4).numpy()

    # 画图
    data_group_output = data_group_output.reshape(-1, 4)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax[0, 0].plot(outputs[:, 0], 'r--')
    ax[0, 0].plot(data_group_output[:, 0], 'b--')
    ax[0, 0].set_xlabel('time_step', fontsize=12)
    ax[0, 0].set_ylabel('X', fontsize=12)

    ax[0, 1].plot(outputs[:, 1], 'r--')
    ax[0, 1].plot(data_group_output[:, 1], 'b--')
    ax[0, 1].set_xlabel('time_step', fontsize=12)
    ax[0, 1].set_ylabel('Y', fontsize=12)

    ax[1, 0].plot(outputs[:, 2], 'r--')
    ax[1, 0].plot(data_group_output[:, 2], 'b--')
    ax[1, 0].set_xlabel('time_step', fontsize=12)
    ax[1, 0].set_ylabel('Vx', fontsize=12)

    ax[1, 1].plot(outputs[:, 3], 'r--')
    ax[1, 1].plot(data_group_output[:, 3], 'b--')
    ax[1, 1].set_xlabel('time_step', fontsize=12)
    ax[1, 1].set_ylabel('Vy', fontsize=12)
    plt.suptitle('Car ID: ' + str(data_car_id[seq_number]), fontsize=20)
    plt.show()

    print('test')


if __name__ == "__main__":
    main()

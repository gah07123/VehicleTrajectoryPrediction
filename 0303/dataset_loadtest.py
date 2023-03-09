import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

train_input = np.array(readtxt("training_data_input.txt"))
test_input = np.array(readtxt("testing_data_input.txt"))
data_input = np.concatenate((train_input, test_input), axis=0)
data_input = data_input.reshape(data_input.shape[0] * data_input.shape[1], -1)

train_input_seq = np.zeros(0)
train_output_seq = np.zeros(0)

for i in range(data_input.shape[0] - 10):
    train_input_seq = np.append(train_input_seq, data_input[i:i + 9])
    train_output_seq = np.append(train_output_seq, data_input[i + 10])

train_input_seq = train_input_seq.reshape(-1, 9)
train_output_seq = train_output_seq.reshape(-1, 1)








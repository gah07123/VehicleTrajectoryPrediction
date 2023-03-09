import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


hidden_size = 16

# batch_size = 1
class Net(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, num_layers=1, batch_first=True):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [1, seq, h] -> [seq, h]
        out = out.view(-1, self.hidden_size)
        out = self.linear(out)  # [seq, h] -> [seq, 1]
        out = out.unsqueeze(dim=0)  # ->[1, seq, 1]
        return out, hidden_prev


start = np.random.randint(3, size=1)[0]
num_time_steps = 50
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

model = Net(hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

hidden_prev = torch.zeros(1, 1, hidden_size)
for iter in range(6000):
    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("iter:{} loss:{}".format(iter, loss.item()))

predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    # 使用预测出来的点继续预测下一个点
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=60)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()
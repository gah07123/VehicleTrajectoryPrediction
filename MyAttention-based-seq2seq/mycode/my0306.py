import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input):
        # input: (batch_size, seq_len, input_size)
        # h0, c0: (num_layers, batch_size, hidden_size)
        batch_size = input.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)
        output, (hn, cn) = self.lstm(input, (h0, c0))
        # output: (batch_size, seq_len, hidden_size)
        return hn, cn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hn, cn):
        # input: (batch_size, 1, input_size)
        # hn, cn: (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(input, (hn, cn))
        # output: (batch_size, 1, hidden_size)
        output = self.fc(output)
        # output: (batch_size, 1, output_size)
        return output, hn, cn

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, output_size, num_layers)

    def forward(self, input, target):
        # input: (batch_size, seq_len, input_size)
        # target: (batch_size, seq_len, output_size)
        batch_size = input.size(0)
        seq_len = target.size(1)
        output = torch.zeros(batch_size, seq_len, target.size(-1)).to(input.device)
        hn, cn = self.encoder(input)
        # hn, cn: (num_layers, batch_size, hidden_size)
        decoder_input = input[:, -1, :].unsqueeze(1)
        # decoder_input: (batch_size, 1, input_size)
        for i in range(seq_len):
            decoder_output, hn, cn = self.decoder(decoder_input, hn, cn)
            output[:, i, :] = decoder_output.squeeze(1)
            decoder_input = target[:, i, :].unsqueeze(1)
            # decoder_input: (batch_size, 1, input_size)
        # output: (batch_size, seq_len, output_size

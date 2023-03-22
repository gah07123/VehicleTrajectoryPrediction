import torch
import torch.nn as nn
import random
import torch.nn.functional as F


# 编码器(LSTM)
class Seq2SeqEncoder(nn.Module):
    '''
    batch_first = True
    '''

    def __init__(self, input_size, enc_hid_size, dec_hid_size, num_layers=1, dropout=0.0):
        super(Seq2SeqEncoder, self).__init__()
        self.input_size = input_size
        self.enc_hid_size = enc_hid_size
        self.dec_hid_size = dec_hid_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.enc_hid_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(enc_hid_size, dec_hid_size)

    def forward(self, input):
        '''
        :param input: [bs, seq len, input size]
        :return: 各个时刻的隐藏状态值output_states = [bs, seq len, enc hid size]
                和最后时刻的隐藏状态值final_h = [num_layers, bs, enc hid size]
        '''
        self.batch_size = input.shape[0]

        output_states, (final_h, final_c) = self.lstm(input)
        final_h = torch.tanh(self.fc(final_h))

        return output_states, final_h


# 注意力机制
class Seq2SeqAttentionMechanism(nn.Module):
    # 实现attention
    def __init__(self, enc_hid_size, dec_hid_size):
        super(Seq2SeqAttentionMechanism, self).__init__()

        self.attn = nn.Linear(enc_hid_size + dec_hid_size, dec_hid_size)
        self.v = nn.Linear(dec_hid_size, 1, bias=False)

    def forward(self, decoder_state_t, encoder_outputs):
        # decoder_state_t : [num_layers, batch size, dec hid size]
        # encoder_states: [batch size, seq_length, enc hid size]
        bs, seq_length, enc_hid_size = encoder_outputs.shape
        # 只取传过来最后一层的隐藏状态
        decoder_state_t = decoder_state_t[-1, :, :]

        # 去掉num_layers维
        # decoder_state_t = decoder_state_t.squeeze(0) # 这一句在batch size = 1 时会报错

        # 扩中间一维 decoder_state_t -> [bs, 1, hidden_size]
        decoder_state_t = decoder_state_t.unsqueeze(1)
        # 使其维度和encoder_states一样
        decoder_state_t = torch.tile(decoder_state_t, dims=(1, seq_length, 1))

        # 用于注意力机制计算的energy
        energy = torch.tanh(self.attn(torch.cat((decoder_state_t, encoder_outputs), dim=2)))
        # energy: [batch_size, seq_length, dec_hid_size]

        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, seq_length]

        return F.softmax(attention, dim=1)


# 解码器
class Seq2SeqDecoder(nn.Module):
    # batch first = True
    def __init__(self, input_size, enc_hid_size, dec_hid_size, output_size, attention, num_layers=1, dropout=0.0):
        super(Seq2SeqDecoder, self).__init__()

        self.input_size = input_size
        self.enc_hid_size = enc_hid_size
        self.dec_hid_size = dec_hid_size
        self.output_size = output_size
        self.num_layers = num_layers
        # lstm层的输入是过了注意力机制之后的context与decoder隐藏层输出concat之后的值，所以input size是enc hid size+input size
        self.lstm = nn.LSTM(input_size=self.enc_hid_size + self.input_size, hidden_size=self.dec_hid_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)

        self.attention = attention
        # [enc hid size + dec hid size + input size -> output size]
        self.fc_out = nn.Linear(self.enc_hid_size + self.dec_hid_size + self.input_size, self.output_size)

    def forward(self, input, hidden, c, encoder_outputs):
        # input = [batch size, 1, input size]
        # hidden = [num_layers, batch size, dec hid size]
        # encoder_outputs = [batch size, seq length, enc hid size]

        bs, seq_length, enc_hid_size = encoder_outputs.shape

        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, seq len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, seq len]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid size]

        lstm_input = torch.cat((input, weighted), dim=2)
        # lstm_input = [batch size, 1, enc hid size * 2]
        output, (hidden, final_c) = self.lstm(lstm_input, (hidden, c))
        # output = [batch size, seq len, dec hid size * n directions]
        # hidden = [n layers * n directions, batch size, dec hid size]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [batch size, 1, dec hid size]
        # hidden = [1, batch size, dec hid size]

        # assert (output == hidden)

        # all change to [batch size, xxx size]
        input = input.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, input), dim=1))
        # predicion = [batch size, output size]
        return prediction, hidden, final_c

# 模型
class Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, seq len, input size]
        # trg = [batch size, trg len, input size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size, seq_len, input_size = src.shape
        trg_len = trg.shape[1]
        trg_output_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_output_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # encoder_outputs = [batch size, seq len, enc hid size], hidden = [num_layers, batch_size, enc hid size]

        # hidden = hidden.squeeze(0)
        # hidden = [batch size, enc hid size]

        # first input to the decoder is 0 = [batch size, 1, input size]
        input = torch.zeros(batch_size, 1, input_size).to(self.device)
        c = torch.zeros(self.encoder.num_layers, batch_size, self.decoder.dec_hid_size).to(self.device)
        # c = [num_layers, batch_size, dec hid size]

        for t in range(0, trg_len):
            # insert input, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, final_c = self.decoder(input, hidden, c, encoder_outputs)
            # output = [batch size, output size], hidden = [num_layers, batch size, dec hid size]

            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next value as next input
            # if not, use predicted value
            input = trg[:, t, :] if teacher_force else output
            # input = [batch size, input size]
            input = input.unsqueeze(1)
            # input = [batch size, 1, input size]

        return outputs

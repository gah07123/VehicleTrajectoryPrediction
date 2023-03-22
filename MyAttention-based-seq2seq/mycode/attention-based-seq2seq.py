import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from pytorch.data_process.dataprocess import *


max_length = 5


# 编码器(LSTM)
class Seq2SeqEncoder(nn.Module):
    '''
    batch_first = True
    '''

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)

    def forward(self, input):
        '''
        :param input: 3d tensor [bs, seq, hs]
        :return: 各个时刻的隐藏状态值output_states [bs,seq,hs]和最后时刻的隐藏状态值final_h [bs,1,hs]
        '''
        self.batch_size = input.shape[0]

        output_states, (final_h, final_n) = self.lstm(input)
        return output_states, final_h


# 注意力机制
class Seq2SeqAttentionMechanism(nn.Module):
    # 实现dot-product的attention
    def __init__(self):
        super(Seq2SeqAttentionMechanism, self).__init__()

    def forward(self, decoder_state_t, encoder_states):
        # decoder_state_t: [bs, hidden_size] 因为后面用的lstmcell，输出维度会少一维
        # encoder_states: [bs, seq_length, hidden_size]
        bs, seq_length, hidden_size = encoder_states.shape
        # 扩中间一维 decoder_state_t -> [bs, 1, hidden_size]
        decoder_state_t = decoder_state_t.unsqueeze(1)
        # 使其维度和encoder_states一样，后续点乘需要 decoder_state_t -> [bs, seq_length, hidden_size]
        decoder_state_t = torch.tile(decoder_state_t, dims=(1, seq_length, 1))
        # 用于注意力机制计算的中间score
        # decoder_state_t与encoder_states按元素相乘,按最后一维hidden_size(就是时间维)求和压缩, dim=-1
        score = torch.sum(decoder_state_t * encoder_states, dim=-1)  # [bs, seq_length]
        # dim=-1, 过softmax, attn_prob代表当前decoder_state与encoder中每一个时刻的关系的权重
        attn_prob = F.softmax(score, dim=-1)  # [bs, seq_length]

        # 权重与encoder_states相乘，最后扩了一维->[bs, seq_length, 1]，并且用了broadcast机制
        # 然后在时间维进行求和压缩
        context = torch.sum(attn_prob.unsqueeze(-1) * encoder_states, dim=1)  # [bs, hidden_size]

        return attn_prob, context


# 解码器
class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, start_id=0, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        # 由于把context和decoder的输出h拼起来之后再给入全连接层，所以hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, output_size)
        # 实例化注意力机制
        self.attention_mechanism = Seq2SeqAttentionMechanism()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.start_id = start_id

    def forward(self, shifted_target, encoder_outputs):
        # 训练阶段调用 teacher forcing训练
        bs, target_length, input_size = shifted_target.shape
        bs, seq_length, hidden_size = encoder_outputs.shape

        output = torch.zeros(bs, target_length, self.output_size)
        probs = torch.zeros(bs, target_length, seq_length)
        # 遍历整个序列
        for t in range(target_length):
            decoder_input_t = torch.zeros(bs, 1, input_size)
            # 如果t为0，则输入初始值为零的h_t,c_t以及零向量decoder_input_t
            # 如果t不为0，则输入上一时刻的h_t,c_t以及shifted_target给该时刻的lstmcell
            if t == 0:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else:
                decoder_input_t = shifted_target[:, t, :]
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))
            # 将给时刻的h_t和encoder_outputs送入注意力机制中生成attn_prob和context
            attn_prob, context = self.attention_mechanism(h_t, encoder_outputs)
            # 拼接context和h_t作为decoder_output [bs, hidden_size * 2]
            decoder_output = torch.cat((context, h_t), dim=-1)
            # 将decoder_output送入全连接层，获得t时刻的output: [bs, output_size] 和权重值attn_prob
            output[:, t, :] = self.fc(decoder_output)
            probs[:, t, :] = attn_prob

        return output, probs

    def inference(self, encoder_outputs):
        # 推理调用
        h_t = None
        results = []
        length = 0
        bs, seq_length, hidden_size = encoder_outputs.shape
        # decoder_input_prev = torch.zeros(bs, 1, hidden_size)
        # output:[bs, output_size]
        output = torch.zeros(bs, self.output_size)

        while True:
            # 初始化一个零向量作为解码器的初始输入，input_t: [bs, input_size]
            decoder_input_t = torch.zeros(bs, self.input_size)
            if h_t is None:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else:
                # 下一时刻的输入是上一时刻的输出h_t
                decoder_input_t = h_t
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))
            # h_t作为查询向量，输入注意力机制输出attn_prob和context
            attn_prob, context = self.attention_mechanism(h_t, encoder_outputs)
            # context和h_t concat之后输入全连接层，decoder_output: [bs, hidden_size * 2]
            decoder_output = torch.cat((context, h_t), -1)
            # 过全连接层，得到需要的输出，output: [bs, output_size]
            output = self.fc(decoder_output)
            results.append(output)

            length = length + 1
            if length == max_length:
                print("stop decoding!")
                break

        prediction = torch.stack(results, dim=0)
        return prediction


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = Seq2SeqEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = Seq2SeqDecoder(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    # 前向传播，训练用
    def forward(self, input_sequence, shifted_target):
        encoder_states, final_h = self.encoder(input_sequence)
        probs, output = self.decoder(shifted_target, encoder_states)

        return probs, output

    # 推理
    def inference(self, input_sequence):
        encoder_states, final_h = self.encoder(input_sequence)
        prediction = self.decoder(encoder_states)

        return prediction


if __name__ == "__main__":
    # 网络参数
    seq_length = 5
    input_size = 4
    output_size = 2
    num_layers = 1
    hidden_size = 128

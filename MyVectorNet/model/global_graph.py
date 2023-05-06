import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalGraph(nn.Module):
    def __init__(self, input_size,
                 global_graph_width,
                 num_global_layers=1):
        super(GlobalGraph, self).__init__()
        self.input_size = input_size
        self.global_graph_width = global_graph_width

        self.layers = nn.Sequential()

        input_size = self.input_size
        for i in range(num_global_layers):
            self.layers.add_module(
                f'Global_layers_{i}', SelfAttentionFCLayer(input_size, self.global_graph_width)
            )
            input_size = self.global_graph_width

    def forward(self, x, **kwargs):
        for name, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttentionFCLayer):
                x = layer(x, **kwargs)
        return x


class SelfAttentionFCLayer(nn.Module):
    def __init__(self, input_size, global_graph_width):
        super(SelfAttentionFCLayer, self).__init__()
        self.input_size = input_size
        self.graph_width = global_graph_width
        self.q_lin = nn.Linear(input_size, global_graph_width)
        self.k_lin = nn.Linear(input_size, global_graph_width)
        self.v_lin = nn.Linear(input_size, global_graph_width)

    def forward(self, x, valid_lens):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_lens)
        x = torch.bmm(attention_weights, value)
        return x

    @staticmethod
    def masked_softmax(X, valid_len):
        """
        :param X: X:3-D tensor
        :param valid_len: 1-D (n, ) or 2-D (n, 1) tensor
        :return: value of masked_softmax
        """
        if valid_len is None:
            return F.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_len.shape[0] != shape[0]:
                valid_len = torch.repeat_interleave(valid_len, repeats=shape[0], dim=0)
            else:
                valid_len = valid_len.reshape(-1)

            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.zeros_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_len):
                cnt = int(cnt.detach().cpu().numpy())
                # scores mask ?
                mask[batch_id, :, cnt:] = True
                # scores mask ?
                mask[batch_id, cnt:] = True
            X_masked = X.masked_fill(mask, -1e12)
            return F.softmax(X_masked, dim=-1) * (1 - mask.float())



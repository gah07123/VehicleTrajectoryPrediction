import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, bias=True, activation="relu", norm="layer"):
        super(MLP, self).__init__()
        # define the activation function
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        else:
            raise NotImplementedError
        # define the normalization layer
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        """
        论文的结构:两层64个hidden units的MLP,后面接layer_norm和relu
        """
        self.linear1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear1.apply(self._init_weights)
        self.norm1 = norm_layer(hidden_size)
        self.act1 = act_layer(inplace=True)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.linear2.apply(self._init_weights)
        self.norm2 = norm_layer(output_size)
        self.act2 = act_layer(inplace=True)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out

if __name__ == "__main__":
    batch_size = 256
    in_feat = 10
    out_feat = 64
    in_tensor = torch.randn((batch_size, 30, in_feat), dtype=torch.float).cuda()

    mlp = MLP(in_feat, out_feat).cuda()

    out = mlp(in_tensor)
    print(out.size())
    print(mlp)



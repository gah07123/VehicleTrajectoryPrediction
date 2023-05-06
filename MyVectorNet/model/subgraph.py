import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
from MyVectorNet.model.MLP import MLP


class SubGraph(nn.Module):
    """
    Subgraph that compute all vectors in a polyline, and get a polyline-level feature
    """
    def __init__(self, input_size, num_subgraph_layers=3, hidden_size=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.hidden_size = hidden_size
        self.output_size = hidden_size

        self.layer_seq = nn.Sequential()
        """
        根据论文的3.2节，polyline_subgraph的构建是先将所有input node features向量经过一个encoder（MLP），encoder之后再进行max_pooling
        这样就得到了encode之后的向量v1（相当于subgraph的node）, 特征维度是hidden_size
        以及max_pooling之后的向量v2（相当于建模了各个node之间的关系），特征维度是hidden_size
        然后将这个两个向量在特征那个维度进行concat，得到output node features,此时特征维度是 hidden_size * 2
        所以下一层subgraph的input_size需要乘以2
        """
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'GLP_{i}', MLP(input_size, hidden_size, hidden_size))
            # 下一层子图input_size = hidden_size * 2
            input_size = hidden_size * 2
        # 最后用一个线性变换把维度变回来
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, sub_data):
        """
        :param sub_data: [x, y, cluster, edge_index, valid_len]
        :return:
        """
        x = sub_data.x
        sub_data.cluster = sub_data.cluster.long()
        sub_data.edge_index = sub_data.edge_index.long()

        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                sub_data.x = x
                agg_data = max_pool(sub_data.cluster, sub_data)
                # 根据cluster，在特征维度concat
                x = torch.cat([x, agg_data.x[sub_data.cluster]], dim=-1)
        # 最后把维度变回来再加一个max_pooling提取整体特征
        x = self.linear(x)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        x = out_data.x

        assert x.shape[0] % int(sub_data.time_step_len[0]) == 0

        return F.normalize(x, p=2.0, dim=1)  # L2正则化






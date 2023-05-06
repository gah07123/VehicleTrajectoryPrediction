import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch, Data
from MyVectorNet.model.MLP import MLP
from MyVectorNet.model.global_graph import GlobalGraph
from MyVectorNet.model.subgraph import SubGraph
from MyVectorNet.utils.ArgoverseLoader import ArgoverseInMem, GraphData


class VectorNetBase(nn.Module):
    def __init__(self,
                 input_size=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 aux_mlp_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool=False,
                 device=torch.device("cpu")):
        super(VectorNetBase, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width
        self.device = device
        self.output_size = 2
        self.horizon = horizon
        # k是指预测多少条轨迹
        self.k = 1
        self.traj_pred_mlp_width = traj_pred_mlp_width

        # subgraph feature extractor
        self.subgraph = SubGraph(input_size, num_subgraph_layers, subgraph_width)
        # global graph
        """
        根据论文的3.3节，global graph输入的定义是subgraph出来的polyline node features拼接上一个邻接矩阵，
        但是由于使用了全连接图，不需要邻接矩阵
        而全局图是一个self-attention
        """
        self.global_graph = GlobalGraph(self.subgraph.output_size + 2, global_graph_width, num_global_graph_layer)

        # auxiliary recovery mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                MLP(self.global_graph_width, aux_mlp_width, aux_mlp_width),
                nn.Linear(aux_mlp_width, self.subgraph.output_size))

    def forward(self, data):
        """
        :param data: [x, y, cluster, edge_index, valid_len]
        :return:
        """
        batch_size = data.num_graphs
        # 所谓的time_step_len就是当前batch的序列的最大有效长度num_valid_len_max
        time_step_len = data.time_step_len[0].int()
        valid_lens = data.valid_len

        id_embedding = data.identifier
        # 获取子图向量
        sub_graph_out = self.subgraph(data)

        if self.training and self.with_aux:
            """
            1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2)就是生成一个(1,valid_len)的随机数
            从1开始是为了避免把agent节点去掉了，valid_lens - 2是为了避免转化成索引后溢出
            time_step_len * torch.arange(batch_size, device=self.device)这个是根据batch_size和time_step_len进行偏移，
            使randoms_mask落在不同batch中，以使不同的batch被mask，避免一个batch中mask了两个节点
            """
            randoms = 1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2) + \
                    time_step_len * torch.arange(batch_size, device=self.device)
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out[mask_polyline_indices]
            sub_graph_out[mask_polyline_indices] = 0.0

        # reconstruct the batch global interaction graph data
        """
        根据论文3.3节的auxiliary构造方法，输入要添加一个identifier embedding来标识global graph中哪一个node被masked了
        """
        # 因为添加了id_embedding, 所以x的特征维度要+2
        x = torch.cat([sub_graph_out, id_embedding], dim=1).view(batch_size, -1, self.subgraph.output_size + 2)
        valid_lens = data.valid_len

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            if self.with_aux:
                # auxiliary只计算被mask的部分
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return global_graph_out, aux_out, aux_gt

            return global_graph_out, None, None

        else:
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            return global_graph_out, None, None


class VectorNetOriginal(nn.Module):
    def __init__(self,
                 input_size=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool=False,
                 device=torch.device("cpu")):
        super(VectorNetOriginal, self).__init__()
        self.polyline_vec_shape = input_size * (2 ** num_subgraph_layers)
        self.output_size = 2
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.traj_pred_mlp_width = traj_pred_mlp_width
        self.k = 1

        self.device = device

        self.basemodel = VectorNetBase(
            input_size=input_size,
            num_subgraph_layers=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred layer: MLP
        self.traj_pred_layer = nn.Sequential(
            MLP(global_graph_width, self.traj_pred_mlp_width, self.traj_pred_mlp_width),
            nn.Linear(self.traj_pred_mlp_width, self.output_size * self.horizon)
        )

    def forward(self, data):
        """

        :param data:[x, y, cluster, edge_index, valid_len]
        :return:
        """
        # global_feat [batch size, time_step_len, global_graph_width]
        global_feat, aux_out, aux_gt = self.basemodel(data)
        # time_step_len = 0 is agent's trajectory
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_layer(target_feat)

        return {"pred": pred, "aux_out": aux_out, "aux_gt": aux_gt}

    def inference(self, data):
        batch_size = data.num_graphs
        # 由于预测的是x，y的增量，所以需要cumsum还原预测轨迹
        pred_traj = self.forward(data)["pred"].view((batch_size, self.k, self.horizon, 2)).cumsum(2)
        return pred_traj


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    pred_len = 30

    INTERMEDIATE_DATA_DIR = "../data/interm_data"
    dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, "val_intermediate")
    dataset = ArgoverseInMem(dataset_input_path)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=False)

    model = VectorNetOriginal(dataset.num_features, with_aux=True, device=device).to(device)
    model.train()
    for i, data in enumerate(tqdm(data_iter, total=len(data_iter))):
        out = model(data.to(device))
        print("Training")

    model.eval()
    for i, data in enumerate(tqdm(data_iter, total=len(data_iter))):
        out = model(data.to(device))
        print("Evaluation")






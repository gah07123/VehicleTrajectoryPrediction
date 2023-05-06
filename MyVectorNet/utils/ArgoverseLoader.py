import gc
import sys
import os
from copy import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader


# 返回全连接邻接矩阵
def get_fc_edge_index(node_indices):
    """
    :param node_indices: np.array([indices]), the indices of nodes connecting with each other
    :return: edge_index (2, edges)
    """
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy


class GraphData(Data):
    # 为了避免不同batch的节点相互连接，这个函数用于更改batch的拼接模式，inc的返回值表示相应矩阵错位步数，一般用于邻接矩阵
    def __inc__(self, key, value):
        if key == 'edge_index':
            # 邻接矩阵每一个batch错位步数为x的数量，即当前节点的数量
            return self.x.size(0)
        elif key == 'cluster':
            # cluster的错位步数为当前cluster的最大值+1
            return int(self.cluster.max().item()) + 1
        else:
            return 0


class ArgoverseInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # gc.collect()
    # 必须加@property，使其能像属性一样访问

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """transform the raw data and store in GraphData"""
        traj_lens = []
        valid_lens = []
        candidate_lens = []
        for raw_path in tqdm(self.raw_paths, desc='Loading raw data...'):
            raw_data = pd.read_pickle(raw_path)
            # 当前seq的轨迹数量
            traj_num = raw_data['feats'].values[0].shape[0]
            traj_lens.append(traj_num)
            # 当前seq的车道数量
            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
            # 用于masked_softmax，当前seq的总向量数，即当前seq的有效长度
            valid_lens.append(traj_num + lane_num)

            candidate_num = raw_data['tar_candts'].values[0].shape[0]
            candidate_lens.append(candidate_num)
        # 所有seq的最大valid_len，用于所有序列node padding
        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        for idx, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData")):
            raw_data = pd.read_pickle(raw_path)

            # input data
            x, cluster, edge_index, identifier = self.get_x(raw_data)
            # y = [60, 1] 代表x, y的增量,有了原点和增量就有了轨迹
            y = self.get_y(raw_data)
            graph_input = GraphData(
                x=torch.from_numpy(x).float(),
                y=torch.from_numpy(y).float(),
                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index).long(),
                identifier=torch.from_numpy(identifier).float(),  # for identifier embedding

                traj_len=torch.tensor([traj_lens[idx]]).int(),
                valid_len=torch.tensor([valid_lens[idx]]).int(),
                time_step_len=torch.tensor([num_valid_len_max]).int(),

                candidate_len_max=torch.tensor([num_candidate_max]).int(),
                candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                candidate_mask=[],
                candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
                offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
                target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),

                orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
                rot_g2l=torch.from_numpy(raw_data['rot_g2l'].values[0]).float().unsqueeze(0),
                rot_l2g=torch.from_numpy(raw_data['rot_l2g'].values[0]).float().unsqueeze(0),
                seq_id=torch.tensor([int(raw_data['seq_id'])]).int(),
                city_name=raw_data['city']
            )
            data_list.append(graph_input)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int):
        data = super(ArgoverseInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad, dtype=data.cluster.dtype)]).long()
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.identifier.dtype)])

        # pad candidate and candidate gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate[:, :2], torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt,
                                       torch.zeros((num_cand_max - len(data.candidate_gt), 1), dtype=data.candidate_gt.dtype)])
        assert data.cluster.shape[0] == data.x.shape[0], "Error: loader error!"
        return data

    @staticmethod
    def get_x(data_seq: pd.DataFrame):
        """
        feat:[xs, ys, vec_x, vec_y, timestep, traffic_control, turn, is_intersection, polyline_id]
        polyline_id: the polyline id of this node belonging to 用于cluster
        :param data_seq:
        :return:
        """
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        # n条[20, 3]的array
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0].astype(np.bool_)
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        # 用点和向量表示轨迹, 构造所有轨迹的向量,统一向量的格式，所以都加上了车道线有的属性，且全部置0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]  # 点
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]  # 向量
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt  # 用于标记该traj，用于cluster
            # [x_s, y_s, vec_x, vec_y, step, traffic_ctrl, is_turn(2), is_intersect, polyline_id]
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turn = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        # 为了cluster,这里的lane_idcs是跟着计算完上面所有的traj features之后的id，所以id就是在traj的id之后计数，所以要加上traj_cnt
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        # [ctrs(2), vec(2), steps, traffic_ctrl, is_turn(2), is_intersect, lane_idcs]
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turn, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))  # 基于id构造cluster list
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)  # 筛选出当前cluster
            # 根据论文, identifier在auxiliary graph中需要使用
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])  # 筛选出当前cluster的最小坐标作为identifier
            if len(indices) <= 1:
                continue
            else:
                # 根据当前cluster构造当前全连接子图的邻接矩阵
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])

        return feats, cluster, edge_index, identifier

    @staticmethod
    def get_y(data_seq: pd.DataFrame):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        # 返回的是偏移量，用点和向量表示实际未来轨迹
        # 0505 bug:最后一项traj_fut[-1, :]漏了一个冒号, 更正: traj_fut[:-1, :]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)


if __name__ == "__main__":
    INTERM_DATA_DIR = "../data/interm_data"

    for folder in ["train", "val", "test"]:
        dataset_input_path = os.path.join(INTERM_DATA_DIR, f"{folder}_intermediate")

        dataset = ArgoverseInMem(dataset_input_path)
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=False)
        for k in range(1):
            for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
                pass




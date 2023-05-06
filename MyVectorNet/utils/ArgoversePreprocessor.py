import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
import argparse
from torch.utils.data import Dataset, DataLoader
from cubic_spline import Spline2D

# 前处理基类
class Preprocessor(Dataset):
    def __init__(self, root_dir, algo='vectornet', obs_horizon=20, obs_range=30, pred_horizon=30):
        self.root_dir = root_dir
        self.algo = algo
        self.obs_horizon = obs_horizon
        self.obs_range = obs_range
        self.pred_horizon = pred_horizon
        self.split = None

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def process(self, dataframe: pd.DataFrame, seq_id: str, map_feat=True):
        """
        :param dataframe: the dataframe
        :param seq_id: str, the sequence id
        :param map_feat: output map feature or not
        :return: DataFrame[(same as original)]
        """
        raise NotImplementedError

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        """
        :param dataframe:
        :param map_feat: output map feature or not
        :return: DataFrame[(same as original)]
        """
        raise NotImplementedError

    def encode_feature(self, *feats):
        """
        :param feats: Dataframe, the data frame containing the filtered data
        :return: DataFrame[POLYLINE_FEATURES, GT, TRAJ_ID_TO_MASK, LANE_ID_TO_MASK, TARJ_LEN, LANE_LEN]
        """
        raise NotImplementedError

    def save(self, dataframe: pd.DataFrame, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe:
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(dataframe, pd.DataFrame):
            return
        if not dir_:
            dir_ = os.path.join(os.path.split(self.root_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        fname = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, fname))

    def process_and_save(self, dataframe: pd.DataFrame, seq_id, dir_=None, map_feat=True):
        """
        process and save
        :param dataframe:
        :param seq_id:
        :param dir_:
        :param map_feat:
        :return:
        """
        df_processed = self.process(dataframe, seq_id, map_feat)
        self.save(df_processed, seq_id, dir_)
        return []

    # 均匀采样
    @staticmethod
    def uniform_candidate_sampling(sampling_range, rate=30):
        """
        uniformly sampling of the target candidate
        :param sampling_range: int the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        :return: rate^2 candidata samples
        """
        x = np.linspace(-sampling_range, sampling_range, rate)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)

    # 对车道线进行三次样条插值之后均匀采样
    def lane_candidate_sampling(self, centerline_list, orig, distance=0.5):
        candidates=[]
        for lane_id, line in enumerate(centerline_list):
            sp = Spline2D(x=line[:, 0], y=line[:, 1])
            s_o, d_0 = sp.calc_frenet_position(orig[0], orig[1])
            s = np.arange(s_o, sp.s[-1], distance)
            ix, iy = sp.calc_global_position_online(s)
            candidates.append(np.stack([ix, iy], axis=1))
        candidates = np.unique(np.concatenate(candidates), axis=0)

        return candidates

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]
            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy


class ArgoversePreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="vectornet",
                 obs_horizon=20,
                 obs_range=100,
                 pred_horizon=30,
                 normalized=True,
                 save_dir=None):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#FF0000", "OTHERS": "#00FFFF", "AV": "#00FF00"}

        self.split = split
        self.normalized = normalized

        self.am = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(os.path.join(self.root_dir, self.split))

        self.save_dir = save_dir

    def __getitem__(self, idx):
        # 用argoverse的api获取文件路径并处理保存
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)

        df = copy.deepcopy(seq.seq_df)
        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    def process(self, dataframe: pd.DataFrame, seq_id, map_feat=True):
        # 获取数据
        data = self.read_argo_data(dataframe)
        # 获取object features
        data = self.get_obj_feats(data)
        # 获取车道等向量用于GNN构建
        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id
        # viz for debug
        # self.visualize_data(data)
        # 按照data中的数据顺序排列返回dataframe
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        city = df["CITY_NAME"].values[0]

        """TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME"""
        # 对agent时间戳进行排序
        agent_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        # 获取50个轨迹点的时间戳,将其按照时间顺序转换为时间步数组steps
        for i, ts in enumerate(agent_ts):
            mapping[ts] = i
        # 获取所有obj的X，Y作为轨迹点列表
        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), axis=1)
        # 根据时间戳按照时间顺序转换成时间步数组steps，所有轨迹的时间步与step数组对应
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)
        # 获取所有TrackID对应的OBJECT_TYPE，对轨迹进行分类
        # objs的索引方式是(trackid, object_type), 对应元素是该trackid所有轨迹点的索引号
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        # keys数组元素是trackid以及对应的obj_type
        keys = list(objs.keys())
        # obj_type数组元素是所有obj_type
        obj_type = [x[1] for x in keys]
        # 获取agent在obj_type数组对应的index
        agent_idx = obj_type.index('AGENT')
        # 根据agent在obj_type数组对应的index，获取objs中对应的(trackid, obj_type)的所有轨迹点的索引号
        idcs = objs[keys[agent_idx]]
        # 根据轨迹点索引号数组idcs，获取agent所有对应的轨迹点和每个轨迹点对应的time_step
        agent_traj = trajs[idcs]
        agent_step = steps[idcs]

        del keys[agent_idx]  # 删除keys中的agent，避免下面重复遍历agent
        others_trajs, others_steps = [], []
        # 重复上述操作
        for key in keys:
            idcs = objs[key]
            others_trajs.append(trajs[idcs])
            others_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agent_traj] + others_trajs
        data['steps'] = [agent_step] + others_steps
        return data

    @staticmethod
    # 计算旋转矩阵，要注意这里的theta是全局航向角
    def get_rotation_matrix(traj, obs_len):
        if obs_len <= 0:
            return
        orig = traj[obs_len-1, :]
        delta = orig - traj[obs_len-2, :]
        theta = np.arctan2(delta[1], delta[0])
        # global to local rotation matrix
        rot_g2l = np.asarray(
            [[np.cos(theta), np.sin(theta)],
             [-np.sin(theta), np.cos(theta)]], dtype=np.float32)
        # local to global rotation matrix
        rot_l2g = np.asarray(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        return rot_g2l, rot_l2g, theta

    def get_obj_feats(self, data):
        # 获取agent历史轨迹序列的最后一个轨迹点的位置用于新建坐标系归一化
        orig = data['trajs'][0][self.obs_horizon - 1].copy().astype(np.float32)

        # compute the rotation matrix
        if self.normalized:
            rot_g2l, rot_l2g, theta = self.get_rotation_matrix(data['trajs'][0], self.obs_horizon)
        else:
            rot_g2l = np.asarray(
                [[1.0, 0.0],
                 [0.0, 1.0], np.float32])
            rot_l2g = copy.deepcopy(rot_g2l)
            theta = None

        # get the target candidates and candidate gt 获取历史轨迹真值和未来轨迹真值以及轨迹周边车道中心线
        agent_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        agent_traj_fut = data['trajs'][0][self.obs_horizon: self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agent_traj_obs, data['city'], viz=False)

        # rotate the center lines and find the reference center line
        agent_traj_fut = np.matmul(rot_g2l, (agent_traj_fut - orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot_g2l, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T
        # 对车道中心线进行采样
        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0])

        if self.split == "test":
            tar_candts_gt, tar_offset_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agent_traj_fut)
            tar_candts_gt, tar_offset_gt = self.get_candidate_gt(tar_candts, agent_traj_fut[-1])

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot_g2l, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth 注意这里对所有轨迹序列进行了padding，长度没有满30的都填充了0
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool_)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            # collect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool_)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.array(feats, np.float32)  # (30, 3)[obs_x, obs_y, 1.0]
        has_obss = np.array(has_obss, np.float32)  # (30)True
        gt_preds = np.array(gt_preds, np.float32)  # (20, 2)[fut_x, fut_y]
        has_preds = np.array(has_preds, np.float32)  # (20)True

        data['orig'] = orig  # 归一化原点
        data['theta'] = theta  # 航向角
        data['rot_l2g'] = rot_l2g
        data['rot_g2l'] = rot_g2l
        data['feats'] = feats           # observation_trajectory
        data['has_obss'] = has_obss
        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds     # groundtruth_future_trajectory
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offset_gt
        data['ref_ctr_lines'] = splines
        data['ref_ctr_idx'] = ref_idx
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range"""
        """周边特征搜索范围(感受野)是100个像素点(obs_range)"""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)

            centerline = np.matmul(data['rot_g2l'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot_g2l'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1  # 车道线segment数量，即向量数量
            # 下面都是车道线每一个节点的属性，所以要乘一个one向量

            # 以点和向量表征车道中心线，第二个是差值代表特征向量
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)
            # 由于使用了点和向量的表示法，所以下面要乘以一个长度为num_segs的向量
            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        # 所有车道线向量节点数
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, axis=0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)  # 车道中心线
        graph['num_nodes'] = num_nodes  # 附近所有车道的节点数
        graph['feats'] = np.concatenate(feats, 0)  # 车道特征向量
        graph['turn'] = np.concatenate(turn, 0)  # 是否是转弯车道
        graph['control'] = np.concatenate(control, 0)  # 是否有交通管制
        graph['intersect'] = np.concatenate(intersect, 0)  # 是否是交叉路口
        graph['lane_idcs'] = lane_idcs  # 节点所属的lane_id，用于cluster
        return graph

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss'].astype(int)
        preds = data['gt_preds']
        has_preds = data['has_preds'].astype(int)
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj, pred, i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        # plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="../data")
    parser.add_argument("-d", "--dest", type=str, default="../data")
    args = parser.parse_args()

    raw_dir = os.path.join(args.root, "raw_data")
    interm_dir = os.path.join(args.dest, "interm_data")

    for split in ["train", "val", "test"]:
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        loader = DataLoader(argoverse_processor,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):
            continue



import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from MyVectorNet.trainer.trainer import Trainer
from MyVectorNet.model.vectornet import VectorNetBase, VectorNetOriginal
from MyVectorNet.utils.loss import VectorLoss
from MyVectorNet.utils.optim_schedule import ScheduledOptim
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.map_representation.map_api import ArgoverseMap


class VectorNetTrainer(Trainer):
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer = 1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas = (0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch = 15,
                 lr_update_freq = 5,
                 lr_decay_rate = 0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):

        super(VectorNetTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            save_folder=save_folder,
            verbose=verbose,
            enable_log=True
        )

        self.horizon = horizon
        self.aux_loss = aux_loss

        # input dim: (20, 10); output dim: (30, 2)
        model_name = VectorNetOriginal
        self.model = model_name(
            self.trainset.num_features,
            self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )
        self.criterion = VectorLoss(aux_loss, reduction="sum")

        # init optimizer
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, "m")
        # load checkpoint
        elif ckpt_path:
            self.load(ckpt_path, "c")

        self.model = self.model.to(self.device)

        # record the init learning rate
        self.write_log("LR", self.lr, 0)

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                                   epoch, 0.0, avg_loss),
            total=len(dataloader)
        )

        for i, data in data_iter:
            n_graph = data.num_graphs
            data = data.to(self.device)

            if training:
                self.optim_schedule.zero_grad()
                loss = self.compute_loss(data)

                loss.backward()
                self.optim.step()
                self.write_log("Train Loss, ", loss.detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    loss = self.compute_loss(data)

                    self.write_log("Eval Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            # print log info
            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                loss.item() / n_graph,
                avg_loss / num_sample
            )
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            learning_rate = self.optim_schedule.step_and_update_lr()
            print(f"Learning rate: {learning_rate:.4}")
            self.write_log("LR", learning_rate, epoch)

        return avg_loss / num_sample

    def compute_loss(self, data):
        out = self.model(data)
        y = data.y.view(-1, self.horizon * 2)
        return self.criterion(out["pred"], y, out["aux_out"], out["aux_gt"])

    # TODO
    def test(self, miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=False,
             save_pred=False):
        """
        :param miss_threshold: float, the threshold for the miss rate, default 2.0m
        :param compute_metric: bool, whether compute the metric
        :param convert_coordinate: bool, True:under original coordinate, False: under the relative coordinate
        :param plot:
        :param save_pred: None
        :return:
        """
        self.model.eval()
        am = ArgoverseMap()
        forecasted_trajectories, gt_trajectories = {}, {}
        city_names = {}
        k = self.model.k
        horizon = self.model.horizon

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                # cumsum:按(axis=1)方向逐个累加到后面的元素上
                # 即把x，y的增量还原成轨迹
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

                origs = data.orig.numpy()
                rots = data.rot_l2g.numpy()
                seq_ids = data.seq_id.numpy()
                city_name = data.city_name

                if gt is None:
                    compute_metric = False

                out = self.model.inference(data.to(self.device))
                dim_out = len(out.shape)
                # pred_y = [batch_size, k, horizon, 2]
                pred_y = out.unsqueeze(dim_out).view((batch_size, k, horizon, 2)).cpu().numpy()

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    seq_id = seq_ids[batch_id]
                    # depends on convert or not(bool: convert_coordinate)
                    forecasted_trajectories[seq_id] = [self.convert_coord(pred_y_k, origs[batch_id], rots[batch_id])
                                                       if convert_coordinate else pred_y_k for pred_y_k in pred_y[batch_id]]
                    gt_trajectories[seq_id] = self.convert_coord(gt[batch_id], origs[batch_id], rots[batch_id]) \
                        if convert_coordinate else gt[batch_id]
                    city_names[seq_id] = city_name[batch_id].values[0]

            # compute the metric
            if compute_metric:
                metric_results = get_displacement_errors_and_miss_rate(
                    forecasted_trajectories,
                    gt_trajectories,
                    k,
                    horizon,
                    miss_threshold
                )
                print("[VectorNet]: The test result: {}".format(metric_results))

            # plot the result
            if plot:
                fig, ax = plt.subplots()
                for key in forecasted_trajectories.keys():
                    # ax.set_xlim(-15, 15)
                    ax.plot(gt_trajectories[key][:, 0], gt_trajectories[key][:, 1], c="#00FF00")
                    ax.plot(forecasted_trajectories[key][0][:, 0], forecasted_trajectories[key][0][:, 1], c="#FF0000")
                    # draw the lanes around the agent
                    self.draw_lane_and_centerline(am, ax, traj=gt_trajectories[key],
                                                  x=gt_trajectories[key][0, 0], y=gt_trajectories[key][0, 1],
                                                  s_range=25.0, city_name=city_names[key])
                    plt.pause(3)
                    ax.clear()

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot_l2g):
        traj_converted = np.matmul(rot_l2g, traj.T).T + orig.reshape(-1, 2)
        return traj_converted

    @staticmethod
    def draw_lane_and_centerline(am, ax, traj, x, y, s_range: float, city_name: str):
        # 车道线
        nearby_lane_id = am.get_lane_ids_in_xy_bbox(query_x=x, query_y=y, city_name=city_name,
                                                    query_search_range_manhattan=s_range)
        # traj_lane_id = am.get_lane_segments_containing_xy(traj[0, 0], traj[0, 1], city_name=city_name)
        # _, _, traj_centerline = am.get_nearest_centerline(np.array([traj[0, 0], traj[0, 1]]), city_name=city_name)

        ax.plot(x, y, '-o', c='r')  # start point here
        for i in range(len(nearby_lane_id)):
            lane_polygons = am.get_lane_segment_polygon(nearby_lane_id[i], city_name=city_name)
            centerline = am.get_lane_segment_centerline(nearby_lane_id[i], city_name=city_name)
            ax.plot(lane_polygons[:, 0], lane_polygons[:, 1], c='gray')
            # ax.plot(centerline[:, 0], centerline[:, 1], '--', c='black')






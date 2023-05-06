import pandas as pd
import os
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap


def compute_features_for_one_seq(traj_df: pd.DataFrame, am: ArgoverseMap, obs_len: int=20, lane_radius: int=5,
                                 obj_radius: int=10, query_bbox=[-100, 100, -100, 100]) -> []:
    """
    return lane & track features
    :param traj_df: trajectory dataframe
    :param am: Argoverse map api
    :param obs_len: 20
    :param lane_radius:
    :param obj_radius:
    :param query_bbox:
    :return:
        agent_features_ls:[x_s, y_s, x_e, y_e object_type, timestamp, track_id, groundtruth_trajectory]
        obj_features_ls:[x_s, y_s, x_e, y_e, object_type, timestamp, track_id]
        lane_features_ls:[left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center:(2, )
    """
    # normalize timestamps







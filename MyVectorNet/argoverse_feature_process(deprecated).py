import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", type=str, default="./data")
    parser.add_argument("-s", "--savedir", type=str, default="./data")
    args = parser.parse_args()

    raw_dir = os.path.join(args.datadir, "raw_data")
    interm_dir = os.path.join(args.savedir, "interm_data")

    am = ArgoverseMap()
    for folder in os.listdir(args.datadir):
        afl = ArgoverseForecastingLoader(os.path.join(args.datadir, folder))
        print(f'folder: {folder}')
        norm_center_dict = {}
        for name in tqdm(afl.seq_list):
            afl_ = afl.get(name)
            path, name = os.path.split(name)
            name, ext = os.path.splitext(name)

            # compute features for one sequence TODO

            # encoding features TODO

            # save features TODO

        with open(os.path.join(args.savedir, f"{folder}-norm_center_dict.pkl"), "wb") as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)








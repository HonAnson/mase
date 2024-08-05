import logging
import os
from os import PathLike
import toml
import torch
import numpy as np
import json

from fvcore.common.config import CfgNode
from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input

# from .search_space import get_search_space_cls
# from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator


from naslib.utils import get_zc_benchmark_api, get_dataset_api
from naslib.utils import get_train_val_loaders, get_project_root
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
)
from naslib.predictors import ZeroCost
from naslib.search_spaces.core import Metric


# For training meta-proxy network
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from einops import rearrange
# from torch import optim
# import torch.nn.functional as F
# import time


logger = logging.getLogger(__name__)


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_proxy_config(config):
    try:
        proxy_config = config["proxy"]
        op_config = proxy_config["op_config"]
        proxies = proxy_config["proxies"]
        dataset_info = proxy_config["proxy_dataset"]
        search_space_info = proxy_config["search_space"]

        return (
            op_config["num_samples"],
            proxies["proxy_list"],
            dataset_info["dataset"],
            search_space_info["search_space"],
        )

    except:
        logger.info("Invalid Config!")
        exit()


def proxy(config: dict | PathLike):

    if not isinstance(config, dict):
        config = load_config(config)
    op_config, proxy_config, dataset_info, search_space_info = parse_proxy_config(
        config
    )  # op_config = list of integers , proxy_config = list of strings

    dataset_info = "cifar10"
    # accepted_dataset = ["cifar10", "cifar100", "ImageNet16-120"]
    # if search_space_info == "nas101" or search_space_info == "nas301":
    #     dataset_info = "cifar10"
    # elif search_space_info == "nas201":
    #     if dataset_info not in accepted_dataset:
    #         dataset_info = "cifar10"

    switch = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}

    n_classes = switch[dataset_info]

    ### Prepare dataloader for running proxies
    config_dict = {
        "dataset": dataset_info,  # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
        "data": str(get_project_root())
        + "/data",  # path to naslib/data where cifar is saved
        "search": {
            "seed": 9001,  # Seed to use in the train, validation and test dataloaders
            "train_portion": 0.7,  # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
            "batch_size": 32,  # batch size of the dataloaders
        },
    }

    dataset_config = CfgNode(config_dict)
    train_loader, val_loader, test_loader, train_transform, valid_transform = (
        get_train_val_loaders(dataset_config)
    )
    scores = {}
    #
    while len(list(scores.keys())) < op_config:
        # Generate models
        # if search_space_info == "nas101":
        #     graph = NasBench101SearchSpace(n_classes)
        if search_space_info == "nas201":
            graph = NasBench201SearchSpace(n_classes)
        elif search_space_info == "nas301":
            graph = NasBench301SearchSpace(n_classes)
        graph.sample_random_architecture(None)
        graph.parse()
        op = graph.get_hash()
        if str(op) not in scores:
            scores[str(op)] = {}
            for zc_proxy in proxy_config:
                zc_predictor = ZeroCost(method_type=zc_proxy)
                score = zc_predictor.query(graph=graph, dataloader=train_loader)
                scores[str(op)][zc_proxy] = score

    # Path for saving the scores
    file_path = "../nas_results/proxy_scores.json"
    # Write dictionary to JSON file
    with open(file_path, "w") as json_file:
        json.dump(scores, json_file)

    # Calculate stddev and mean for future data normalisation
    proxy_mean_stddev = {}
    for zc_proxy in proxy_config:
        temp = []
        proxy_mean_stddev[zc_proxy] = {}
        for key in scores:
            score = scores[key][zc_proxy]
            temp.append(score)
        temp = np.array(temp)
        mean = np.mean(temp)
        stddev = np.std(temp)
        if stddev == 0:
            stddev = 1e-8
        proxy_mean_stddev[zc_proxy]["mean"] = mean
        proxy_mean_stddev[zc_proxy]["stddev"] = stddev

    # Save mean and standard deviation of proxy score distribution
    file_path = "../nas_results/proxy_mean_stddev.json"
    # Write dictionary to JSON file
    with open(file_path, "w") as json_file:
        json.dump(proxy_mean_stddev, json_file)
    return




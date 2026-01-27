import sys
import logging
from logging import getLogger
import os

import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer
from recbole.utils import (
    init_logger,
    get_flops,
    get_environment,
    init_seed,
    set_color
)
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.fearec import FEARec

from model import DiSAM4Rec  


if __name__ == '__main__':

    config = Config(model=DiSAM4Rec, config_file_list=['config.yaml'])

    time_flag = "_TIME" if "use_time_embedding" in config and config["use_time_embedding"] else ""
    if "use_revit_distill" in config and config["use_revit_distill"]:
        distill_layers_str = "-".join(map(str, config['distill_layers']))
        model_name = f"SIGMA_ReViT_L{distill_layers_str}_T{config['distill_temperature']}_lam{config['lambda_distill']}{time_flag}"
    else:
        model_name = f"SIGMA{time_flag}"

    if config['anti_sparsity']:
        model_name += f"_SPARSE{config['remain_ratio']}"

    config["model"] = model_name

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir = os.path.join(root_dir, 'log', model_name)
    checkpoint_dir = os.path.join(root_dir, 'saved_model', model_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config['logger'] = True
    config['save_log'] = True
    config['log_dir'] = log_dir

    config['save_model'] = True
    config['checkpoint_dir'] = checkpoint_dir
    config['checkpoint_file'] = os.path.join(checkpoint_dir, f"{model_name}.pth")
    config['checkpoint_save_mode'] = 'best'

    init_logger(config)
    logger = getLogger()
    logger.info(f"Model: {model_name}")
    logger.info(f"Log Dir: {log_dir}")
    logger.info(f"Checkpoint Dir: {checkpoint_dir}")
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(f"Dataset: {dataset}")
    if config['anti_sparsity'] and 0 < config['remain_ratio'] < 1.0:
        import numpy as np

        total_interactions = len(dataset.inter_feat)
        keep_num = int(total_interactions * config['remain_ratio'])

        np.random.seed(config['seed'])
        keep_indices = np.random.choice(total_interactions, keep_num, replace=False)
        keep_indices = sorted(keep_indices)
        inter_df = pd.DataFrame(dataset.inter_feat)

        pruned_inter_df = inter_df.iloc[keep_indices].reset_index(drop=True)

        dataset.inter_feat = type(dataset.inter_feat)(pruned_inter_df)

        logger.info(f"[Anti-Sparsity] Retained {keep_num} / {total_interactions} interactions")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    local_rank = config.local_rank if hasattr(config, 'local_rank') else 0
    init_seed(config['seed'] + local_rank, config['reproducibility'])
    model = DiSAM4Rec(config, dataset).to(config['device'])
    logger.info(f"Model architecture:\n{model}")

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    if "use_revit_distill" in config and config["use_revit_distill"]:
        logger.info(set_color("[ReViT Distillation ENABLED]", "green"))
        logger.info(f"  distill_layers = {config['distill_layers']}")
        logger.info(f"  distill_temperature = {config['distill_temperature']}")
        logger.info(f"  lambda_distill = {config['lambda_distill']}")
        distill_method = config["distill_method"] if "distill_method" in config else "mse"
        logger.info(f"  distill_method = {distill_method}")

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    environment_tb = get_environment(config)
    logger.info("Environment:\n" + environment_tb.draw())

    logger.info(set_color("Best validation result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test result", "yellow") + f": {test_result}")

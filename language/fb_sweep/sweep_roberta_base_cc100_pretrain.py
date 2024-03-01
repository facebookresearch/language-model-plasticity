# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sweep
import datetime
from sweep import hyperparam
import argparse


def get_grid(args):

    max_update = 125000 # number of gradient updates
    global_batch_size = 2048 
    local_batch_size = 32 # per gpu batch size

    num_gpus = args.num_gpus * args.num_nodes
    update_freq = global_batch_size // (num_gpus*local_batch_size)
    assert local_batch_size * update_freq * num_gpus == global_batch_size

    return [
        hyperparam("--arch", "roberta_base", save_dir_key=lambda val: val),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--fast-stat-sync"),
        hyperparam("--fp16"),
        hyperparam("--num-workers", 2),
        hyperparam("--task", "masked_lm"),
        hyperparam("--criterion", "masked_lm"),
        hyperparam("--sample-break-mode", "complete"),
        hyperparam("--shorten-method", "truncate"),
        hyperparam("--shorten-data-split-list", "train"),
        hyperparam("--tokens-per-sample", 512), # window size
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--adam-eps", 1e-6),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 7e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", 10000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--dropout", 0.1),
        hyperparam("--attention-dropout", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--batch-size", local_batch_size, save_dir_key=lambda val: f"ms{val}"), # batch-size 32
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--validate-interval-updates", 500),
        hyperparam("--save-interval-updates", 500), # checkpoints every 500 updates
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--keep-interval-updates-pattern", max_update // 10),
        hyperparam("--no-epoch-checkpoints"),  # only save checkpoints based on num steps
        hyperparam("--keep-best-checkpoints", 0), 

        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 50),
        hyperparam("--wandb-project", "plasticity"), # wandb project name 
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Pretrain')
    parser.add_argument('--choice', default='monolingual')
    parser.add_argument('--partition', default='learnlab')
    args = parser.parse_args()

    choice = args.choice
    partition = args.partition
    
    def add_extra_options_func(parser):
        # parser.add_argument('--ddp-backend', default='legacy_ddp')
        return None

    TIME = datetime.datetime.today().strftime("%Y%m%d-%H%M")
    if choice == 'monolingual':
        CORPUS = 'cc100' 
        STAGE = 'pretrain'
        EXP_DIR = './plasticity/exps'
        DATA_DIR = './plasticity/datasets/{}/en/bin'.format(CORPUS)
    elif choice == 'multilingual_cc100_mini_amount_of_en':
        CORPUS = 'cc100_mini'
        STAGE = 'pretrain'
        EXP_DIR = './plasticity/exps'
        DATA_DIR = './plasticity/datasets/cc100/{}/bin_amount_en'.format(CORPUS)
    else:
        print('Choice wrong')
        exit
        
    scheduler_args = ["--prefix=plasticity", 
                      "--num-gpus=8",
                      "--num-nodes=4",
                      "--checkpoints-dir={}/{}/{}{}/".format(EXP_DIR, CORPUS, STAGE, TIME),
                      "--partition={}".format(args.partition), # Specify your partition
                      "--time=2160",
                      "--data={}".format(DATA_DIR),
                      "--num-trials=-1", 
                      "--no-tensorboard",
                      "--constraint=volta32gb" # Use V100 32 GB
                       ]
    sweep.main(get_grid, postprocess_hyperparams, 
               add_extra_options_func=add_extra_options_func, # hyper-params in train.py
               scheduler_args=scheduler_args # hyper-params in the sweep.py
               )
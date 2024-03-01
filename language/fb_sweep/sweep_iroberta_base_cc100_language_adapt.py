# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sweep
from sweep import hyperparam

import datetime
import argparse

def get_grid(args):
    """the args is amended by add_extra_options"""

    max_update = 125000
    global_batch_size = 2048
    local_batch_size = 32

    num_gpus = args.num_gpus * args.num_nodes
    update_freq = global_batch_size // (num_gpus*local_batch_size)
    assert local_batch_size * update_freq * num_gpus == global_batch_size

    hp = [
        # hyperparam("--train-subset", "train" if not args.local else "valid"), #!!!!!!!!!!!!!
        # hyperparam("--combine-valid-subsets"),
        hyperparam("--arch", "firoberta_base", save_dir_key=lambda val: val),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--fast-stat-sync"),
        hyperparam("--fp16"),
        hyperparam("--num-workers", 2),
        hyperparam("--task", "masked_lm"),
        hyperparam("--criterion", "masked_lm"),

        hyperparam("--sample-break-mode", "complete"),
        hyperparam("--shorten-method", "truncate"), #????
        hyperparam("--shorten-data-split-list", "train"), # ????

        hyperparam("--tokens-per-sample", 512),
        hyperparam("--optimizer", args.optimizer, save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--adam-eps", 1e-6),
        hyperparam("--clip-norm", 0.0),
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
        hyperparam("--save-interval-updates", 500),
        hyperparam("--keep-interval-updates", 250), # save last 250 checkpoints!!!!!!!!
        # hyperparam("--keep-interval-updates-pattern", max_update // 10),
        hyperparam("--no-epoch-checkpoints"),  # only save checkpoints based on num steps
        hyperparam("--keep-best-checkpoints", 0),  # don't save checkpoint_best.pt

        hyperparam("--finetune-from-model", args.restore_file),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 50),
        hyperparam("--wandb-project", "forgeT"),
        hyperparam("--clear_embed_every_K_updates", 1250000) # never reset
    ]
    if args.optimizer == 'adamef':
        hp.append(hyperparam("--lr-emb", args.lr_emb))
    return hp


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Language Adaptation')
    parser.add_argument('--lang', default='de')
    parser.add_argument('--bin_dir', default='bin')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--restore_file', 
                        default="None")
    args = parser.parse_args()

    cmd_optimizer = args.optimizer
    cmd_restore_file = args.restore_file

    def add_extra_options_func(parser): # return a parser for usage in get_grid function
        # parser.add_argument('--ddp-backend', default='legacy_ddp')
        parser.add_argument('--optimizer', default=cmd_optimizer)
        parser.add_argument('--restore-file', default=cmd_restore_file)
        if cmd_optimizer == 'adamef':
            parser.add_argument('--lr-emb', default=7e-4)
        return None

    TIME = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    CORPUS = 'cc100' 
    STAGE = 'adapt'
    EXP_DIR = '/checkpoint/yhc/inductivise-lm/inductivise-lm/exps'
    DATA_DIR = '/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/{}/{}/{}'.format(CORPUS, args.lang, args.bin_dir)

    scheduler_args = ["--prefix=forgeT", 
                      "--num-gpus=8",
                      "--num-nodes=4",
                      "--checkpoints-dir={}/{}/{}{}/".format(EXP_DIR, CORPUS, STAGE, TIME),
                      "--partition=learnlab",
                      "--time=2160",
                      "--data={}".format(DATA_DIR),
                      "--num-trials=-1", 
                      "--no-tensorboard",
                      "--constraint=volta32gb"
                       ]
    sweep.main(get_grid, postprocess_hyperparams, 
               add_extra_options_func=add_extra_options_func, # arguments not in the sweep.py but in train.py
               scheduler_args=scheduler_args # arguments in the sweep.py
               )
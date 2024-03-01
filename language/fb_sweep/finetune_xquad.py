# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import datetime
import argparse
import os
import ast
import sys
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


FAIRSEQ_DIR = '/checkpoint/yhc/inductivise-lm'


def main():
    parser = argparse.ArgumentParser(description='Finetune a fairseq model in SQUAD')
    parser.add_argument('--sentencepiece-model', required=True)
    parser.add_argument('--checkpoint', required=True, help='The model to finetune')

    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--valid', default='squad_valid_en', help='Validation data') #!!!
    parser.add_argument('--valid-gold',
                        default='/checkpoint/namangoyal/storage/xlm_roberta/dataset/squad/dev-v1.1.json', #!!!!
                        help='Gold validation data')
    #/checkpoint/kmarchisio/intern-proj/data/MLQA_XQUAD_indiv/bin/

    parser.add_argument('--destdir', default='/checkpoint/yhc/inductivise-lm/inductivise-lm/exps/qa-{}'.format(datetime.datetime.today().strftime("%Y%m%d-%H%M")))
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (defaults to 10)')
    parser.add_argument('--arch', default='roberta_base', help='Model architecture (defaults to roberta_base)')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--partition', choices=['learnlab', 'devlab'], default='learnlab')
    
    parser.add_argument('--local', action='store_true')

    parser.add_argument('--freeze_token_emb', type=ast.literal_eval, default=False, help='freeze embeddings')
    parser.add_argument('--freeze_lm_head', type=ast.literal_eval, default=False, help='freeze lm head')
    parser.add_argument('--freeze_body', type=ast.literal_eval, default=False, help='freeze body')
    
    args = parser.parse_args()

    def get_grid(_):
        grid = []

        max_epoch = args.epochs

        # Tried these, 0.00003 was best.
        ## peak_lr = [0.00001, 0.00003, 0.00005, 0.0001, 0.0003, 0.0005]
        peak_lr = [0.00003]

        warmup_ratios = [0.06]
        max_sentences = [24]

        total_num_updates = (160000 // max_sentences[0] ) * max_epoch
        skip_long = False
        lr_scheduler = 'poly'

        truncate_sequence = False
        shuffle = True
        pooler_dropout = None

        init_token = 0
        sep_token = 2

        grid += [
            hyperparam('--finetune-from-model', args.checkpoint),
            hyperparam('--bpe', 'sentencepiece'),
            hyperparam('--sentencepiece-model', args.sentencepiece_model),

            hyperparam('data', args.data_folder, positional_arg=True), # folder for train/valid subsets
            hyperparam('--data-files', [args.valid_gold]), #!!!!!!!!!!!!!?????
            hyperparam("--valid-subset", args.valid),
            hyperparam("--train-subset", 'train_en'),
            hyperparam("--gen-subset", args.valid), # data subset to generate (train, valid, test)

            hyperparam("--dataset-impl", "mmap"),
            hyperparam("--num-workers", 4),

            hyperparam("--no-save-optimizer-state"),
            hyperparam("--no-last-checkpoints"),
            hyperparam("--best-checkpoint-metric", 'qna_extractive_loss'),
            # hyperparam('--maximize-best-checkpoint-metric', [False], binary_flag=True), #!!!!

            hyperparam("--update-freq", 1),
            hyperparam("--max-update", total_num_updates),
            
            hyperparam("--task", "squad", save_dir_key=lambda val: 'squad'),
            hyperparam('--criterion', 'squad'),
            hyperparam("--max-tokens", 4400),
            hyperparam("--max-sentences", max_sentences),
            hyperparam("--max-positions", 512),
            hyperparam("--arch", args.arch, save_dir_key=lambda val: val),
        
            hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
            hyperparam("--adam-betas", "(0.9, 0.98)"),
            hyperparam("--adam-eps", 1e-6),
            hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),

            hyperparam("--dropout", 0.1),
            hyperparam("--attention-dropout", 0.1),
            hyperparam("--activation-dropout", 0.0),
            hyperparam("--weight-decay", 0.01),

            # TODO: Maybe remove these. Commented-out in Jonas'.
            hyperparam("--required-batch-size-multiple", 1),
            ## hyperparam("--init-token", 0),
            ## hyperparam("--separator-token", 2), 

            hyperparam("--ddp-backend", "c10d"),
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 100),
            hyperparam("--seed", args.seed, save_dir_key=lambda val: f"seed{val}"),
            hyperparam("--wandb-project", "forgeT"),
            # hyperparam("--validate-interval-updates", 5),

        ]

        if skip_long:
            grid += [
                hyperparam("--skip-invalid-size-inputs-valid-test"),
            ]

        if not shuffle:
            grid += [
                hyperparam("--no-shuffle"),
            ]

        if pooler_dropout is not None:
            grid += [
                hyperparam("--pooler-dropout", pooler_dropout, save_dir_key=lambda val: f"poolerdr{val}"),
            ]

        if truncate_sequence:
            grid += [
                hyperparam("--truncate-sequence")
            ]

        # TODO: Why is this commented out?
        # if num_classes is not None:
        #     grid += [
        #         hyperparam('--num-classes', num_classes),
        #     ]

        # lr scheduler
        if lr_scheduler == 'poly':
            warmup_updates = [int(total_num_updates * x) for x in warmup_ratios]

            grid += [
                hyperparam("--lr-scheduler", "polynomial_decay"),
                hyperparam("--lr", peak_lr, save_dir_key=lambda val: f"lr{val}"),
                hyperparam("--total-num-update", total_num_updates),
                hyperparam("--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"),
            ]
        else:
            grid += [
                hyperparam("--lr-scheduler", "fixed"),
                hyperparam("--lr", peak_lr, save_dir_key=lambda val: f"lr{val}"),
            ]

        if args.local:
            grid += [
                hyperparam("--log-format", "json"),
                hyperparam("--log-interval", 1),
            ]
        
        if args.arch in ['froberta_base', 'firoberta_base']:
            grid.append(hyperparam('--freeze_token_emb', args.freeze_token_emb)) # comment out this for pure roberta
            grid.append(hyperparam('--freeze_lm_head', args.freeze_lm_head))
            grid.append(hyperparam('--freeze_body', args.freeze_body))
        print(args.freeze_token_emb, args.freeze_lm_head, args.freeze_body)
        if args.freeze_body == False:
            print('False freeze body')
        return grid


    def postprocess_hyperparams(args, config):
        """Postprocess a given hyperparameter configuration."""
        pass

    # Run training
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + [
        '-t', '-1', '-g', '1', '-n', '1', '--constraint', 'volta32gb',
        '--partition', args.partition, '--time', '2880', '--prefix', 'QA', '--checkpoint', args.destdir,
        '--script', FAIRSEQ_DIR + '/train.py',
        '--no-tensorboard',
    ]

    if args.local:
        sys.argv.append('--local')
    sweep.main(get_grid, postprocess_hyperparams)
    sys.argv = original_argv

    # copy dictionary into the data-folder


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    main()

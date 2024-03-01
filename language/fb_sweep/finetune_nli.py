# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import os
import shutil
import subprocess
import datetime
import ast

import sys

import fb_sweep.sweep as sweep
from fb_sweep.sweep import hyperparam

from scripts import nli_utils
from scripts import spm_preprocess

FAIRSEQ_DIR = '/checkpoint/yhc/inductivise-lm/'


def write_lines(lines, path):
    with open(path, mode='w', encoding='utf-8') as f:
        for line in lines:
            print(line, file=f)


def main():
    parser = argparse.ArgumentParser(description='Finetune a fairseq model in NLI')
    parser.add_argument('--sentencepiece-model', required=True)
    parser.add_argument('--checkpoint', required=True, help='The model to finetune')
    parser.add_argument('--train', required=True, help='Training data in jsonl format')
    parser.add_argument('--valid', required=True, help='Validation data in jsonl format')
    parser.add_argument('--train-langs', default=None, nargs='+', help='Filter languages from the training data')
    parser.add_argument('--valid-langs', default=None, nargs='+', help='Filter languages from the valid data')
    parser.add_argument('--destdir',
                        default='/checkpoint/yhc/inductivise-lm/inductivise-lm/exps/nli-{}'.format(datetime.datetime.today().strftime("%Y%m%d-%H%M")))

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (defaults to 1e-5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (defaults to 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (defaults to 10)')
    parser.add_argument('--arch', default='roberta_base', help='Model architecture (defaults to roberta_base)')
    parser.add_argument('--freeze_token_emb', type=ast.literal_eval, default=False, help='freeze embeddings')
    parser.add_argument('--freeze_lm_head', type=ast.literal_eval, default=False, help='freeze lm head')
    parser.add_argument('--freeze_body', type=ast.literal_eval, default=False, help='freeze body')
    parser.add_argument('--max-positions', type=int, default=512, help='Max sequence length (defaults to 512)')
    parser.add_argument('--no-epoch-checkpoints', action='store_true')

    parser.add_argument('--workers', metavar='N', type=int, default=20, help='Number of workers for preprocessing (defaults to 20)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--partition',
        choices=['xlmg', 'devaccel', 'learnaccel', 'scavenge', 'learnlab'],
        default='learnlab')
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.destdir + '/data/raw', exist_ok=True)
    os.makedirs(args.destdir + '/data/bin', exist_ok=True)

    # Extract input/labels
    n_train_samples = None
    for split, path, langs in ('train', args.train, args.train_langs), ('valid', args.valid, args.valid_langs):
        data = nli_utils.read_nli(path, langs=langs)
        original_size = len(data)
        data = [sample for sample in data if sample['gold_label'] != '-']
        l = set([sample['gold_label'] for sample in data])
        print('*' * 80)
        print(split, path, langs)
        print(l)
        print('*' * 80)
        assert all(sample['gold_label'] in ('contradiction', 'entailment', 'neutral') for sample in data)
        filtered_size = len(data)
        if split == 'train':
            n_train_samples = filtered_size
        if filtered_size != original_size:
            print(f'Filtered {filtered_size}/{original_size} samples from {path}', file=sys.stderr)
        for name, field in ('input0', 'sentence1'), ('input1', 'sentence2'), ('label', 'gold_label'):
            write_lines([sample[field] for sample in data], f'{args.destdir}/data/raw/{split}.{name}.txt')
    # Tokenize and binarize input
    for field in 'input0', 'input1':
        spm_preprocess.preprocess(
            spm_model_path=args.sentencepiece_model,
            train_path=f'{args.destdir}/data/raw/train.{field}.txt',
            valid_path=f'{args.destdir}/data/raw/valid.{field}.txt',
            test_path=None,
            dest_dir=f'{args.destdir}/data/bin/{field}',
            # output_format='id',
            output_format='piece',
            workers=args.workers,
        )

    # Binarize labels
    subprocess.run([
        'fairseq-preprocess',
        '--trainpref', f'{args.destdir}/data/raw/train.label.txt',
        '--validpref', f'{args.destdir}/data/raw/valid.label.txt',
        '--only-source',
        '--thresholdsrc', '0',
        '--destdir', f'{args.destdir}/data/bin/label',
        '--workers', str(args.workers),
    ])

    total_updates = args.epochs*(((n_train_samples - 1) // args.batch_size) + 1)

    def get_grid(_):
        hparams = [
            hyperparam('--train-subset', 'train'),

            hyperparam('data', f'{args.destdir}/data/bin', positional_arg=True),

            hyperparam('--no-last-checkpoints'),
            hyperparam('--no-save-optimizer-state'),

            hyperparam('--reset-optimizer'),
            hyperparam('--reset-dataloader'),
            hyperparam('--reset-meters'),

            hyperparam('--best-checkpoint-metric', 'accuracy'),
            hyperparam('--maximize-best-checkpoint-metric', [True], binary_flag=True),

            hyperparam('--restore-file', args.checkpoint),

            hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
            hyperparam('--ddp-backend', 'no_c10d'),
            hyperparam('--num-workers', 1 if not args.local else 0),

            hyperparam('--task', 'sentence_prediction', save_dir_key=lambda val: 'sentpred'),
            hyperparam('--init-token', 0, save_dir_key=lambda val: f'bos{val}'),
            hyperparam('--separator-token', 2, save_dir_key=lambda val: f'sep{val}'),
            hyperparam('--max-positions', args.max_positions),
            hyperparam('--regression-target', [False], binary_flag=True),
            hyperparam('--arch', args.arch, save_dir_key=lambda val: val),
            hyperparam('--criterion', 'sentence_prediction'),
            hyperparam('--num-classes', 3),

            hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
            hyperparam('--adam-betas', '(0.9, 0.98)', save_dir_key=lambda val: 'b2_0.98'),
            hyperparam('--adam-eps', 1e-6, save_dir_key=lambda val: f'eps{val}'),
            hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

            # hyperparam('--lr-scheduler', 'polynomial_decay'),
            hyperparam('--lr', args.lr, save_dir_key=lambda val: f'lr{val}'),
            # hyperparam('--warmup-updates', int(0.06*total_updates), save_dir_key=lambda val: f'wu{val}'), # 03/01/2023
            # hyperparam('--total-num-update', total_updates), # 03/01/2023
            hyperparam('--max-update', total_updates, save_dir_key=lambda val: f'mu{val}'),

            hyperparam('--dropout', 0.1, save_dir_key=lambda val: f'dr{val}'),
            hyperparam('--attention-dropout', 0.1, save_dir_key=lambda val: f'atdr{val}'),
            hyperparam('--weight-decay', 0.01, save_dir_key=lambda val: f'wd{val}'),

            hyperparam('--batch-size', args.batch_size, save_dir_key=lambda val: f'ms{val}'),
            hyperparam('--required-batch-size-multiple', 1),
            hyperparam('--update-freq', 1, save_dir_key=lambda val: f'uf{val}'),

            hyperparam('--seed', args.seed, save_dir_key=lambda val: f's{val}'),
            hyperparam('--log-format', 'json'),
            hyperparam('--log-interval', 25),
            hyperparam("--wandb-project", "forgeT-20230704"),
        ]
        if args.arch in ['froberta_base', 'firoberta_base']:
            hparams.append(hyperparam('--freeze_token_emb', args.freeze_token_emb)) # comment out this for pure roberta
            hparams.append(hyperparam('--freeze_lm_head', args.freeze_lm_head))
            hparams.append(hyperparam('--freeze_body', args.freeze_body))
        print(args.freeze_token_emb, args.freeze_lm_head, args.freeze_body)
        if args.freeze_body == False:
            print('False freeze body')

        if args.no_epoch_checkpoints:
            hparams.append(hyperparam('--no-epoch-checkpoints'))
        return hparams

    def postprocess_hyperparams(args, config):
        pass

    # Run training
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + [
        '-t', '-1', '-g', '1', '-n', '1', '--constraint', 'volta32gb',
        '--partition', args.partition, '--time', '2880', '--prefix', 'NLI', '--checkpoint', args.destdir,
        '--script', FAIRSEQ_DIR + '/train.py',
        '--no-tensorboard',
    ]
    if args.local:
        sys.argv.append('--local')
    sweep.main(get_grid, postprocess_hyperparams)
    sys.argv = original_argv

    # Copy dictionaries into the checkpoint directory
    for directory in next(os.walk(args.destdir))[1]:
        if directory != 'data':
            for split in 'input0', 'input1', 'label':
                os.makedirs(f'{args.destdir}/{directory}/{split}', exist_ok=True)
                shutil.copyfile(f'{args.destdir}/data/bin/{split}/dict.txt', f'{args.destdir}/{directory}/{split}/dict.txt')


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    main()

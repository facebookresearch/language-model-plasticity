# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import collections
import numpy as np
import os
import sys
import torch

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
from pathlib import Path

from scripts import nli_utils

from fb_sweep.model_list.NLI_basic import MODELS as MODELS_basic
from fb_sweep.model_list.NLI_thai_ablation import MODELS as MODELS_thai_ablation
from fb_sweep.model_list.NLI_multilingual_pretrain_ablation import MODELS as MODELS_mp


def load_model(config):
    model = get_model(
        model_path=config["model_path"],
        dict_path=config.get("dict_path"),
        # suffix=fairseq_cfg.checkpoint.checkpoint_suffix or "",
        **config.get("model_overrides", {}),
    )
    return model


def get_model(model_path, model_type="RobertaModel", dict_path=None, suffix="", bpe="empty", **kwargs):
    print(model_path)
    assert bpe != "empty", "bpe must be set in model_overrides of the model in model configurations"
    # check if model directory exists
    assert os.path.exists(os.path.dirname(model_path))

    model_path = Path(model_path)
    model_name_or_path = str(model_path.parent)
    checkpoint_file = model_path.name
    data_name_or_path = "."
    if dict_path is not None:
        dict_path = Path(dict_path)
        assert dict_path.exists()
        # HACK: The setup_task method will look in the data_dir for dict.txt
        # https://github.com/pytorch/fairseq/blob/dea66cc294a18dd4d9e59aa0af8d51f951e83884/fairseq/tasks/language_modeling.py#L141
        data_name_or_path = str(dict_path.parent)
    print('*' * 80)
    print(dict_path, data_name_or_path, suffix, checkpoint_file, model_name_or_path)
    model = RobertaModel.from_pretrained(
        model_name_or_path=model_name_or_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path,
        suffix=suffix,
        # If the criterion in the checkpoint is set to
        # vocab_parallel_cross_entropy, then the output layer will skip the
        # gather_from_model_parallel_region call, which is necessary.
        # So here we override the criterion to force the output layer to call
        # gather_from_model_parallel_region.
        criterion="cross_entropy",
        bpe=bpe,
        **kwargs,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--choice', required=True, help='choices of models')
    parser.add_argument('-m', '--model', required=True, help="The name of the model you want to test") 
    parser.add_argument('-d', '--data', required=True, help="The data to evaluate on")
    parser.add_argument('-l', '--langs', nargs='+', default=None)
    parser.add_argument('--out_type', default='int')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    
    if args.choice == '5k': #!!!!!!!!!
        MODELS = MODELS_cc100_5k
    elif args.choice == '125k':
        MODELS = MODELS_cc100
    elif args.choice == 'basic':
        MODELS = MODELS_basic
    elif args.choice == 'xlmr':
        MODELS = MODELS_xlmr
    elif args.choice == 'americas':
        MODELS = MODELS_americas
    elif args.choice == 'thai_ablation':
        MODELS = MODELS_thai_ablation
    elif args.choice == "multilingual_pretrain_ablation":
        MODELS = MODELS_mp
    config = MODELS[args.model]
    data = nli_utils.read_nli(args.data, args.langs)

    print('*' * 80)
    print(args.langs)
    model = load_model(config) 
    if args.fp16:
        model.half()
    model.cuda()
    model.eval()
    
    label_fn = config.get('label_fn', lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial]))
    head = config.get('head', 'sentence_classification_head')
    print('Loaded model {}'.format(config), file=sys.stderr)
    sample = data[0]
    toks = [model.encode(sample['sentence1'], sample['sentence2'], out_type=args.out_type)[:model.model.max_positions()] for sample in data]
    print(model.encode(sample['sentence1'], out_type=args.out_type), model.encode(sample['sentence2'], out_type=args.out_type))
    # print(model.model.max_positions())
    print(model.task.dictionary)
    print('Done binarizing')

    lens = np.array([x.numel() for x in toks])
    sorted_inds = np.argsort(-lens)

    predictions = [None] * len(data)
    max_tokens = 4*model.model.max_positions()
    oom = False
    i = 0
    while i < len(data):
        j = min(len(data), i + (max_tokens // lens[sorted_inds[i]]))
        batch = collate_tokens(
            [toks[ind] for ind in sorted_inds[i:j]],
            pad_idx=model.task.dictionary.pad()
        )
        # print(f'{i}-{j} ({lens[sorted_inds[i]]})', file=sys.stderr)
        try:
            with torch.no_grad():
                batch_predictions = model.predict(head, batch).argmax(dim=1).tolist()
        except RuntimeError as e:
            if 'CUDA out of memory' not in str(e):
                raise e
            max_tokens = max_tokens // 2
            oom = True
            print(f'Reducing max tokens to {max_tokens} due to OOM', file=sys.stderr)
            continue

        for prediction, ind in zip(batch_predictions, sorted_inds[i:j]):
            predictions[ind] = label_fn(prediction)
        if not oom:
            max_tokens *= 2
            print(f'Increasing max tokens to {max_tokens}', file=sys.stderr)
        i = j

    lang2correct = collections.defaultdict(int)
    lang2total = collections.defaultdict(int)
    print('*' * 80)
    pred_dist = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
    for sample, prediction in zip(data, predictions):
        # print(prediction, sample['gold_label'])
        lang = sample.get('language') or 'en'
        if sample['gold_label'] == prediction:
            lang2correct[lang] += 1
        lang2total[lang] += 1
        pred_dist[prediction] +=1
    print(pred_dist)

    langs = sorted(lang2total.keys())
    print('\t'.join(langs + ['avg']))
    accuracies = [lang2correct[lang] / lang2total[lang] for lang in langs]
    accuracies.append(sum(lang2correct.values()) / sum(lang2total.values()))
    print('\t'.join([f'{100*acc:.1f}' for acc in accuracies]))


if __name__ == '__main__':
    main()

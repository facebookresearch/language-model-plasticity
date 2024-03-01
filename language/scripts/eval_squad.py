# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

"""
Evaluate the perplexity of a trained language model.

"""

import json
import unicodedata
import string
import collections
import math
import random
from os import path
import os
import subprocess
import tempfile
import torch
import sys
import re
from fairseq import options, progress_bar, tasks, utils, checkpoint_utils
from fairseq.data import encoders
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
import argparse
from pathlib import Path
from omegaconf import OmegaConf,open_dict
from tqdm import *
import gc

import sentencepiece as spm
from fairseq.models.roberta import RobertaModel

MLQA_EVAL_FILE="/checkpoint/yhc/inductivise-lm/scripts/mlqa_evaluation_v1.py"
OTHER_EVAL_FILE="/checkpoint/yhc/inductivise-lm/scripts/evaluate-v1.1.py"


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, bpe, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    # tokenizer = BasicTokenizer(do_lower_case=True)
    # tok_text = " ".join(tokenizer.tokenize(orig_text))
    tok_text = bpe.decode(bpe.encode(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # if verbose_logging:
        # print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        # print("Length not equal after stripping spaces: '%s' vs '%s'",
        #               orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        # if verbose_logging:
        # print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def postprocess_model_output(
        example_index_to_features,
        dictionary,
        language,
        args,
        example_index,
):
    bpe = spm.SentencePieceProcessor(model_file=args.sentencepiece_model)
    # all_predictions = collections.OrderedDict()

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "token_to_orig_map", "text"]
    )
    # for id in example_index_list:
    prelim_predictions = []
    assert example_index in example_index_to_features, "id missing {0}".format(example_index)
    for span_idx, unique_example in enumerate(example_index_to_features[example_index]):
        start_logits, end_logits, text, mask, orig, idx_map, token_is_max_context = unique_example
        padding_mask = text.eq(dictionary.pad()) #TODO: Use real BPE dictionary?
        mask[padding_mask] = 0
        start_indexes = _get_best_indexes(start_logits, args.n_best_size)
        end_indexes = _get_best_indexes(end_logits, args.n_best_size)
        token_to_orig_map = {}
        mnz = mask.nonzero()[0]
        start_idx = mnz[0]
        end_idx = mnz[-1]
        assert (len(mnz) == len(idx_map))
        for j in range(start_idx, end_idx + 1):
            token_to_orig_map[j] = idx_map[j - start_idx]
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(text) or start_index < start_idx or start_index > end_idx:
                    continue
                if end_index >= len(text) or end_index < start_idx or end_index > end_idx:
                    continue
                if end_index < start_index:
                    continue
                if not token_is_max_context[start_index]:
                    continue
                length = end_index - start_index + 1
                if length > 30:
                    continue
                # tok_tokens = task.dictionary.string(text[start_index:(end_index + 1)])
                # assert ('<pad>' not in tok_tokens)
                # tok_text = encoder.decode_lines([tok_tokens])[1][0]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=id,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index],
                        token_to_orig_map=token_to_orig_map,
                        text=text))

    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
    doc_tokens = orig.split()
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= args.n_best_size:
            break
        if pred.start_index > 0:
            tok_tokens = dictionary.string(pred.text[pred.start_index:(pred.end_index + 1)])
            tok_tokens = [int(i) for i in tok_tokens.split()]
            if '<unk>' in tok_tokens:
                tok_tokens = re.sub("<unk>", "", tok_tokens)
            # From https://github.com/huggingface/transformers/blob/81d6841b4be25a164235975e5ebdcf99d7a26633/src/transformers/data/metrics/squad_metrics.py
            # based on this question: https://github.com/huggingface/transformers/issues/3510
            tok_text = bpe.decode(tok_tokens)
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())

            if len(tok_text) == 0:
                continue
            if tok_text[0] == " ":
                tok_text = tok_text[1:]

            if language == 'zh':
                final_text = tok_text
            else:
                # NOTE: I think get_final_text fails b/c we're using spm with IDs
                # Look into later.
                orig_doc_start = pred.token_to_orig_map[pred.start_index]
                orig_doc_end = pred.token_to_orig_map[pred.end_index]
                orig_tokens = " ".join(doc_tokens[orig_doc_start:(orig_doc_end + 1)])
                final_text = get_final_text(tok_text, orig_tokens, bpe)

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True
        nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
    assert len(nbest) >= 1
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if best_non_null_entry is None:
            if entry.text:
                best_non_null_entry = entry
    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    assert len(nbest_json) >= 1
    # all_predictions[id] = nbest_json[0]["text"]
    # print('return')
    return (example_index, nbest_json[0]["text"])


def eval_dataset(
        task,
        model,
        dataset_name,
        data_file,
        args,
        prefix='test',
        use_cuda=True,
        language='en',
):
    # NOTE: From Kelly - IDK why these were commented out.  Jonas used his own
    # argparse instead of the squad parser (idk why):
    # args.min_len = None
    # args.max_len = None
    # args.keep_empty = False
    itr = task.get_batch_iterator(
        dataset=task.datasets[args.gen_subset],
        max_tokens=(args.max_tokens or 4096) * 2,
        max_sentences=48,
        max_positions=4096,
        seed=args.seed,
        ignore_invalid_inputs=False,
    ).next_epoch_itr(shuffle=False)

    was_training = model.training
    model.eval()

    example_index_to_features = collections.defaultdict(list)
    total = 0
    with torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for index, batch in tqdm(enumerate(t)):
            if use_cuda:
                batch = utils.move_to_cuda(batch)
            # Dummy batch
            if 'net_input' not in batch:
                continue
            start_res, end_res, _ = model(
                **batch['net_input'],
                classification_head_name="squad_span_classification_head",
                paragraph_mask=batch['paragraph_mask'],
            )[0]
            for start_logits, end_logits, id, text, mask, orig, idx_map, token_is_max_context in zip(start_res, end_res, batch['squad_ids'],
                                                            batch['net_input']['src_tokens'],
                                                            batch['paragraph_mask'],
                                                            batch['actual_txt'],
                                                            batch['idx_map'],
                                                            batch['token_is_max_context']):
                # example_index_to_features[id].append((start_logits, end_logits, text, mask, orig, idx_map, token_is_max_context))
                example_index_to_features[id].append(
                    (start_logits.float().cpu().data.numpy(), end_logits.float().cpu().data.numpy(), text.cpu(), mask.cpu().data.numpy(), orig,
                    idx_map.cpu().data.numpy(), token_is_max_context.cpu().data.numpy())
                )
                total += 1

    ##########################
    # MAYBE(todo): Maybe put back multiprocessing if I want to later (it had a cuda
    # error that i'll have to debug):
    # https://discuss.pytorch.org/t/multiprocessing-for-cuda-error/16866/8
    ## p = Pool(8)
    ## partial_f = partial(postprocess_model_output, example_index_to_features, task.dictionary, args)
    ## example_index_list = deepcopy(list(example_index_to_features.keys()))
    ## return_values = p.map(partial_f, example_index_list)
    ## p.close()

    return_values = []
    example_index_list = deepcopy(list(example_index_to_features.keys()))
    print('Total Examples:', len(example_index_list))
    for i, example_index in enumerate(example_index_list):
        return_values.append(postprocess_model_output(example_index_to_features,
            task.dictionary, language, args, example_index))
        if i % 10 == 0:
            print(i, '...', end=' ', flush=True)
    print('\n')
    ##########################

    all_predictions = collections.OrderedDict()
    for k, v in return_values:
        all_predictions[k] = v

    fname = path.join(
        args.save_dir,
        "{0}_{1}_{2}.json".format(
            dataset_name,
            prefix,
            random.randint(0, 100000)
        )
    )
    with open(fname, 'w') as f:
        json.dump(all_predictions, f)
        f.flush()

    try:
        # if 'mlqa' in dataset:
        if 'mlqa' in args.gen_subset:
            res = subprocess.check_output(
                [
                    "python",
                    MLQA_EVAL_FILE,
                    data_file,
                    f.name,
                    language
                ],
                cwd=path.dirname(path.realpath(__file__))
            )
        else:
            res = subprocess.check_output(
                [
                    "python",
                    OTHER_EVAL_FILE,
                    data_file,
                    f.name,
                ],
                cwd=path.dirname(path.realpath(__file__))
            )
    except subprocess.CalledProcessError as e:
        res = e.output
    print(dataset_name, res.decode('utf-8'))

    if was_training:
        model.train()
    return res



def main(parsed_args):

    print(parsed_args)

    use_cuda = torch.cuda.is_available() # and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(parsed_args.path.split(':'),
                                                        task=task,
                                                        arg_overrides={"bpe": "sentencepiece",
        "sentencepiece_model": parsed_args.sentencepiece_model, 'remove_sentence_classification_head': False})

    assert len(models) == 1

    model = models[0]
    if use_cuda:
        model.cuda()

    for arg in vars(parsed_args).keys():
        if arg in args:
            setattr(args, arg, getattr(parsed_args, arg))
        else:
            with open_dict(args):
                setattr(args, arg, getattr(parsed_args, arg))

    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    eval_dataset(task, model, parsed_args.gen_subset, args.gold_data_file, args, language=args.language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate SQuAD, XQuAD, MLQA')
    parser.add_argument('--path', type=str, required=True, help='Path to model to eval.')
    parser.add_argument('--sentencepiece-model', type=str, required=True, help='Path to bpe/spm.')
    parser.add_argument('-d', '--data', default='/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/XQUAD_MLQA/bin')
    parser.add_argument('--gold_data_file', required=True) # parser.add_argument('--gold_data_file', default='/checkpoint/namangoyal/storage/xlm_roberta/dataset/MLQA/MLQA_V1/dev/dev-context-en-question-en.json')
    parser.add_argument('--batch_size',  default=48)
    parser.add_argument('--gen_subset', required=True) # parser.add_argument('--gen_subset',  default='mlqa_valid_en')
    parser.add_argument('-l', '--language',  default='en')
    parser.add_argument('--save_dir', default='/checkpoint/yhc/inductivise-lm/inductivise-lm/results/cc100/xquad_mlqa/')
    parser.add_argument('--log-format',  default='json')
    parser.add_argument('-m',  type=str)

    parser.add_argument('--n-best-size', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--max-query-length', type=int, default=64)
    parser.add_argument('--max-positions', type=int, default=512)
    parser.add_argument('--max_tokens', type=int, default=4400)
    parser.add_argument('--model-dim', type=int, default=1024)
    parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                        help='add prev_output_tokens to sample, used for encoder-decoder arch')
    parser.add_argument('--squad-validation-updates', type=int, default=-1)

    options.add_common_eval_args(parser)
    args = options.parse_args_and_arch(parser)
    args.task = 'squad'
    main(args)
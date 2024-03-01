# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import collections
import json
import os
import sys

from scripts import m2m


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f.read().splitlines()]


def write_jsonl(data, path):
    with open(path, mode='x', encoding='utf-8') as f:
        for sample in data:
            print(json.dumps(sample), file=f)


def read_nli(path, langs=None):
    data = read_jsonl(path)
    if langs is not None:
        data = [sample for sample in data if sample.get('language', 'en') in langs]
    
    lang2count = collections.defaultdict(int)
    for sample in data:
        lang2count[sample.get('language', 'en')] += 1
    
    if langs:
        assert set(lang2count.keys()) == set(langs)
    
    nlangs = len(lang2count)
    assert nlangs > 0
    lens = list(lang2count.values())
    assert all([lens[0] == length for length in lens])

    print(f'Loaded {lens[0]} samples in {nlangs} languages from {path}', file=sys.stderr)
    return data


def convert_xtreme(original_path, xtreme_path, output_path):
    data = read_nli(original_path)
    entry2translation = {}
    langs = set([sample['language'] for sample in data])
    langs.remove('en')
    for lang in langs:
        with open(xtreme_path.replace('{LANG}', lang), encoding='utf-8') as f:
            lines = [line for line in f.read().split('\n') if line]
            for line in lines:
                cols = line.split('\t')
                assert len(cols) == 5
                entry = (lang, cols[0].strip(), cols[1].strip(), cols[4])  # (language, sentence1, sentence2, label)
                translation = (cols[2].strip(), cols[3].strip())  # (sentence1, sentence2)
                entry2translation[entry] = translation
    for sample in data:
        entry = (sample['language'], sample['sentence1'].strip(), sample['sentence2'].strip(), sample['gold_label'])
        if sample['language'] == 'en':
            continue
        translation = entry2translation[entry]
        sample['sentence1'] = translation[0]
        sample['sentence2'] = translation[1]
        del sample['sentence1_tokenized']
        del sample['sentence2_tokenized']
    write_jsonl(data, output_path)


def bleu(path, ref_lang='en'):
    import sacrebleu
    import numpy as np
    data = read_nli(path)
    langs = sorted(set([sample['language'] for sample in data]))
    lang2id2sent = {lang: dict() for lang in langs}
    for sample in data:
        lang2id2sent[sample['language']][f"{sample['pairID']}-sentence1"] = sample['sentence1'].strip()
        lang2id2sent[sample['language']][f"{sample['pairID']}-sentence2"] = sample['sentence2'].strip()
    lang2sents = {lang: [] for lang in langs}
    for lang in langs:
        for name in sorted(lang2id2sent[ref_lang].keys()):
            lang2sents[lang].append(lang2id2sent[lang][name])
    bleus = [sacrebleu.corpus_bleu(lang2sents[lang], [lang2sents[ref_lang]]).score for lang in langs]
    bleus.append(np.mean(bleus))
    print('\t'.join(langs + ['avg']))
    print('\t'.join([f'{bleu:.1f}' for bleu in bleus]))


# def bleu(path, ref_lang='en'):
#     import sacrebleu
#     import numpy as np
#     import sacremoses
#     normalizer = sacremoses.MosesPunctNormalizer(pre_replace_unicode_punct=True, post_remove_control_chars=True)
#     data = read_xnli(path, langs=['ar', 'bg', 'en'])
#     langs = sorted(set([sample['language'] for sample in data]))
#     lang2id2sent = {lang: dict() for lang in langs}
#     for sample in data:
#         lang2id2sent[sample['language']][f"{sample['pairID']}-sentence1"] = sample['sentence1'].strip()
#         lang2id2sent[sample['language']][f"{sample['pairID']}-sentence2"] = sample['sentence2'].strip()
#     lang2sents = {lang: [] for lang in langs}
#     for lang in langs:
#         for name in sorted(lang2id2sent[ref_lang].keys()):
#             lang2sents[lang].append(lang2id2sent[lang][name])
#     bleus = [sacrebleu.corpus_bleu([normalizer.normalize(sent) for sent in lang2sents[lang]], [lang2sents[ref_lang]]).score for lang in langs]
#     bleus.append(np.mean(bleus))
#     print('\t'.join(langs + ['avg']))
#     print('\t'.join([f'{bleu:.1f}' for bleu in bleus]))


def translate_test_m2m(args, tgt_lang='en'):
    data = read_nli(args.input)
    langs = set([sample['language'] for sample in data])
    langs.remove(tgt_lang)

    lang2sentences = {lang: set() for lang in langs}
    for sample in data:
        lang = sample['language']
        if lang != tgt_lang:
            lang2sentences[lang].update({sample['sentence1'], sample['sentence2']})

    lang2translations = {}
    for lang in langs:
        sentences = lang2sentences[lang]
        translations = m2m.translate(sentences, lang, tgt_lang, args)
        lang2translations[lang] = {src: tgt for src, tgt in zip(sentences, translations)}

    for sample in data:
        lang = sample['language']
        if lang == tgt_lang:
            continue
        sample['sentence1'] = lang2translations[lang][sample['sentence1']]
        sample['sentence2'] = lang2translations[lang][sample['sentence2']]
        del sample['sentence1_tokenized']
        del sample['sentence2_tokenized']
    write_jsonl(data, args.output)


def copy_field(input, output, field, from_lang):
    data = read_nli(input)
    id2val = {sample['pairID']: sample[field] for sample in data if sample['language'] == from_lang}
    for sample in data:
        sample[field] = id2val[sample['pairID']]
    write_jsonl(data, output)


def main():
    parser = argparse.ArgumentParser(description='XNLI tools')
    subparsers = parser.add_subparsers(title='subcommands')

    parser_convert_xtreme = subparsers.add_parser('convert-xtreme')
    parser_convert_xtreme.add_argument('--original', default='data/xnli/official/XNLI-1.0/xnli.test.jsonl')
    parser_convert_xtreme.add_argument('--xtreme', default='data/xnli/xtreme/original/translate-test/XNLI_translate-test_test-{LANG}-en-translated.tsv')
    parser_convert_xtreme.add_argument('-o', '--output', default=sys.stdout.fileno())
    parser_convert_xtreme.set_defaults(subcommand='convert-xtreme')

    parser_bleu = subparsers.add_parser('bleu')
    parser_bleu.add_argument('--data', required=True)
    parser_bleu.set_defaults(subcommand='bleu')

    parser_translate_test_m2m = subparsers.add_parser('translate-test-m2m')
    parser_translate_test_m2m.add_argument('-i', '--input', default=sys.stdin.fileno())
    parser_translate_test_m2m.add_argument('-o', '--output', default=sys.stdout.fileno())
    m2m.add_m2m_args(parser_translate_test_m2m)
    parser_translate_test_m2m.set_defaults(subcommand='translate-test-m2m')

    parser_copy_field = subparsers.add_parser('copy-field')
    parser_copy_field.add_argument('-i', '--input', default=sys.stdin.fileno())
    parser_copy_field.add_argument('-o', '--output', default=sys.stdout.fileno())
    parser_copy_field.add_argument('--field', choices=['sentence1', 'sentence2'])
    parser_copy_field.add_argument('--from-lang', default='en')
    parser_copy_field.set_defaults(subcommand='copy-field')

    args = parser.parse_args()
    if args.subcommand == 'convert-xtreme':
        convert_xtreme(original_path=args.original, xtreme_path=args.xtreme, output_path=args.output)
    if args.subcommand == 'bleu':
        bleu(args.data)
    if args.subcommand == 'translate-test-m2m':
        translate_test_m2m(args)
    if args.subcommand == 'copy-field':
        copy_field(args.input, args.output, args.field, args.from_lang)


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    main()
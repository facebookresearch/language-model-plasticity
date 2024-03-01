# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import sacremoses
try:
    import spacy
    IS_SPACY_AVAILABLE = True
except ImportError:
    IS_SPACY_AVAILABLE = False


M2M_DIR = '/checkpoint/artetxe/models/m2m'
FAIRSEQ_DIR= '/private/home/artetxe/software/fairseq-m2m'
MAX_POSITIONS = 1024
DEFAULT_BEAM=5


DISTRIBUTED_ARGS = [
    '--distributed-world-size', '1', '--distributed-no-spawn',
    '--pipeline-model-parallel',
    '--pipeline-chunks', '1',
]

MODELS = {
    '418m': {
        'path': f'{M2M_DIR}/418M_last_checkpoint.pt',
        'lang_pairs': f'{M2M_DIR}/language_pairs_small_models.txt',
        'default_max_tokens': 16*1024,
        'ngpus': 1,
    },
    '1.2b': {
        'path': f'{M2M_DIR}/1.2B_last_checkpoint.pt',
        'lang_pairs': f'{M2M_DIR}/language_pairs_small_models.txt',
        'default_max_tokens': 16*1024,
        'ngpus': 1,
    },
    '12b-2gpus': {
        'path': f'{M2M_DIR}/12b_last_chk_2_gpus.pt',
        'lang_pairs': f'{M2M_DIR}/language_pairs.txt',
        'default_max_tokens': 4*1024,
        'ngpus': 2,
        'extra_args': DISTRIBUTED_ARGS + [
            '--pipeline-encoder-balance', '[26]',
            '--pipeline-encoder-devices', '[0]',
            '--pipeline-decoder-balance', '[3,22,1]',
            '--pipeline-decoder-devices', '[0,1,0]',
        ]
    },
    '12b-4gpus': {
        'path': f'{M2M_DIR}/12b_last_chk_4_gpus.pt',
        'lang_pairs': f'{M2M_DIR}/language_pairs.txt',
        'default_max_tokens': 4*1024,
        'ngpus': 4,
        'extra_args': DISTRIBUTED_ARGS + [
            '--pipeline-encoder-balance', '[1,15,10]',
            '--pipeline-encoder-devices', '[0,1,0]',
            '--pipeline-decoder-balance', '[3,11,11,1]',
            '--pipeline-decoder-devices', '[0,2,3,0]',
        ]
    },
    '12b-8gpus': {
        'path': f'{M2M_DIR}/12b_last_chk_8_gpus.pt',
        'lang_pairs': f'{M2M_DIR}/language_pairs.txt',
        'default_max_tokens': 4*1024,
        'ngpus': 8,
        'extra_args': DISTRIBUTED_ARGS + [
            '--pipeline-encoder-balance', '[1,6,6,6,7]',
            '--pipeline-encoder-devices', '[0,4,5,1,0]',
            '--pipeline-decoder-balance', '[1,6,6,6,6,1]',
            '--pipeline-decoder-devices', '[0,2,6,7,3,0]',
        ]
    },
}


class SentenceSegmenter(object):

    def __init__(self):
        if not IS_SPACY_AVAILABLE:
            raise Exception('Please install Spacy for sentence segmentation')
        self.model = spacy.load('xx_sent_ud_sm')

    def segment(self, text):
        if not text:
            return []
        sents = map(str, self.model(text).sents)
        return [sent for sent in sents if sent.strip()]


def translate(
    lines, src_lang, tgt_lang, model,
    beam=DEFAULT_BEAM, sampling=False, temperature=None, topp=None,
    pivot_lang=None, pivot_beam=None, pivot_sampling=None, pivot_temperature=None, pivot_topp=None,
    segment_sentences=False,
    max_tokens=None, threads=20, seed=1
):
    if pivot_lang is not None:
        lines = translate(
            lines=lines,
            src_lang=src_lang,
            tgt_lang=pivot_lang,
            model=model,
            beam=beam if pivot_beam is None else pivot_beam,
            sampling=sampling if pivot_sampling is None else pivot_sampling,
            temperature=temperature if pivot_temperature is None else pivot_temperature,
            topp=topp if pivot_topp is None else pivot_topp,
            segment_sentences=segment_sentences,
            max_tokens=max_tokens,
            threads=threads,
            seed=seed,
        )
        src_lang = pivot_lang
        segment_sentences = False

    if max_tokens is None:
        max_tokens = MODELS[model]['default_max_tokens']

    if segment_sentences:
        segmenter = SentenceSegmenter()
        segment_sentences = lambda x: segmenter.segment(x)
    else:
        segment_sentences = lambda x: [x] if x.strip() else []
    
    with tempfile.TemporaryDirectory() as tmp:
        sents_per_line = []
        with open(f'{tmp}/raw.{src_lang}', mode='x', encoding='utf-8') as f:
            for line in lines:
                sents = segment_sentences(line)
                sents_per_line.append(len(sents))
                for sent in sents:
                    print(sent, file=f)

        prev_python_path = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = FAIRSEQ_DIR

        # Tokenize with SentencePiece
        subprocess.run([
            'python3', f'{FAIRSEQ_DIR}/scripts/spm_encode.py',
            '--model', f'{M2M_DIR}/spm.128k.model',
            '--output_format', 'piece',
            '--inputs', f'{tmp}/raw.{src_lang}',
            '--outputs', f'{tmp}/spm.{src_lang}'
        ])
        os.remove(f'{tmp}/raw.{src_lang}')

        # Trim sentences longer than MAX_POSITIONS
        with open(f'{tmp}/spm.{src_lang}', encoding='utf-8', errors='surrogateescape') as fin:
            with open(f'{tmp}/trimmed.{src_lang}', mode='x', encoding='utf-8', errors='surrogateescape') as fout:
                for line in fin:
                    toks = line.rstrip('\n').split()
                    print(' '.join(toks[:MAX_POSITIONS-2]), file=fout)
        os.remove(f'{tmp}/spm.{src_lang}')

        # Binarize
        subprocess.run([
            'python3', f'{FAIRSEQ_DIR}/fairseq_cli/preprocess.py',
            '--source-lang', src_lang, '--target-lang', tgt_lang,
            '--only-source',
            '--testpref', f'{tmp}/trimmed',
            '--thresholdsrc', '0', '--thresholdtgt', '0',
            '--destdir', f'{tmp}/data_bin',
            '--srcdict', f'{M2M_DIR}/data_dict.128k.txt',
            '--tgtdict', f'{M2M_DIR}/data_dict.128k.txt',
            '--workers', str(threads),
        ])
        os.remove(f'{tmp}/trimmed.{src_lang}')

        # Translate
        command = [
            'python3', f'{FAIRSEQ_DIR}/fairseq_cli/generate.py',
            f'{tmp}/data_bin',
            '--max-tokens', str(max_tokens),
            '--path', MODELS[model]['path'],
            '--fixed-dictionary', f'{M2M_DIR}/model_dict.128k.txt',
            '-s', src_lang, '-t', tgt_lang,
            '--remove-bpe', 'sentencepiece',
            '--beam', str(beam),
            '--task', 'translation_multi_simple_epoch',
            '--lang-pairs', MODELS[model]['lang_pairs'],
            '--decoder-langtok', '--encoder-langtok', 'src',
            '--gen-subset', 'test',
            '--dataset-impl', 'mmap',
            '--fp16',
            '--seed', str(seed),
            '--results-path', f'{tmp}/output',
        ]
        command += MODELS[model].get('extra_args', [])
        if sampling:
            command.append('--sampling')
        if temperature is not None:
            command += ['--temperature', str(temperature)]
        if topp is not None:
            command += ['--sampling-topp', str(topp)]
        subprocess.run(command)

        if prev_python_path is None:
            del os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = prev_python_path

        ind2translation = {}
        with open(f'{tmp}/output/generate-test.txt', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('H-'):
                    name, _, translation = line.split('\t')
                    ind = int(name[2:])
                    ind2translation[ind] = translation
        sent_translations = [ind2translation[i] for i in range(len(ind2translation))]
        normalizer = sacremoses.MosesPunctNormalizer(lang=tgt_lang)
        sent_translations = [normalizer.normalize(translation) for translation in sent_translations]
        assert len(sent_translations) == sum(sents_per_line)
        translations = []
        start_ind = 0
        for i in sents_per_line:
            translations.append(' '.join(sent_translations[start_ind:start_ind+i]))
            start_ind += i
        return translations


def main():
    parser = argparse.ArgumentParser(description='Translate using M2M')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno())
    parser.add_argument('-o', '--output', default=sys.stdout.fileno())
    parser.add_argument('--src', required=True, help='Source language')
    parser.add_argument('--tgt', required=True, help='Target language')
    parser.add_argument('--pivot', default=None, help='Pivot language (optional)')

    # TODO Add option to parse src lang from jsonl as --src field:FIELD
    parser.add_argument('--jsonl', metavar='FIELD', nargs='+', default=None, help='Input/output in jsonl instead of plain text. Translates the given FIELDs.')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--beam', metavar='N', default=None, const=DEFAULT_BEAM, type=int, nargs='?')
    mode.add_argument('--unconstrained-sampling', metavar='TEMPERATURE', default=None, const=1.0, type=float, nargs='?')
    mode.add_argument('--nucleus-sampling', metavar='TOP_P', default=None, const=0.9, type=float, nargs='?')

    # Parameters for translating into the pivot language. If not provided we use the ones in the other direction.
    pivot_mode = parser.add_mutually_exclusive_group()
    pivot_mode.add_argument('--pivot-beam', metavar='N', default=None, const=DEFAULT_BEAM, type=int, nargs='?')
    pivot_mode.add_argument('--pivot-sampling', metavar='TEMPERATURE', default=None, const=1.0, type=float, nargs='?')
    pivot_mode.add_argument('--pivot-nucleus-sampling', metavar='TOP_P', default=None, const=0.9, type=float, nargs='?')

    parser.add_argument('--model', required=True, choices=MODELS.keys())
    parser.add_argument('--segment-sentences', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--threads', metavar='N', type=int, default=10, help='Number of threads (defaults to 10)')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--jobs', type=int, default=0)
    parser.add_argument('--parallel-jobs', type=int, default=None)
    parser.add_argument('--partition', choices=['xlmg', 'learnaccel', 'devaccel', 'learnfair', 'scavenge'], default='xlmg')
    parser.add_argument('--working', default=str(Path.home() / 'm2m'), help='Working directory (must be accessible from all nodes)')

    args = parser.parse_args()

    os.makedirs(args.working, exist_ok=True)
    args.working += '/{:%Y%m%d%H%M%S}-{}2{}{}'.format(datetime.datetime.now(), args.src, '2' + args.pivot if args.pivot else '', args.tgt)
    os.makedirs(args.working, exist_ok=False)

    # If the input is in jsonl format extract the actual text from the relevant fields
    if args.jsonl is not None:
        jsonl_data = []
        with open(args.input, encoding='utf-8') as fin:
            with open(args.working + '/input.txt', mode='x', encoding='utf-8') as fout:
                for line in fin:
                    sample = json.loads(line)
                    jsonl_data.append(sample)
                    for field in args.jsonl:
                        print(sample[field].strip(), file=fout)
        original_input = args.input
        original_output = args.output
        args.input = args.working + '/input.txt'
        args.output = args.working + '/output.txt'

    if args.beam is None and args.unconstrained_sampling is None and args.nucleus_sampling is None:
        # Default to beam search
        args.beam = DEFAULT_BEAM
    if args.pivot_beam is None and args.pivot_sampling is None:
        # Use the same settings as in the other direction
        args.pivot_beam = args.beam
        args.pivot_unconstrained_sampling = args.unconstrained_sampling
        args.pivot_nucleus_sampling = args.nucleus_sampling

    def translation_task(args, input, output):
        with open(input, encoding='utf-8') as f:
            lines = f.read().splitlines()
        translations = translate(
            lines=lines,
            src_lang=args.src,
            tgt_lang=args.tgt,
            pivot_lang=args.pivot,
            model=args.model,
            beam=1 if args.beam is None else args.beam,
            sampling=args.beam is None,
            temperature=args.unconstrained_sampling,
            topp=args.nucleus_sampling,
            pivot_beam=1 if args.pivot_beam is None else args.pivot_beam,
            pivot_sampling=args.pivot_beam is None,
            pivot_temperature=args.pivot_sampling,
            pivot_topp=args.pivot_nucleus_sampling,
            segment_sentences=args.segment_sentences,
            max_tokens=args.max_tokens,
            threads=args.threads,
            seed=args.seed,
        )
        with open(output, mode='w', encoding='utf-8') as f:
            for translation in translations:
                print(translation, file=f)
    if args.jobs == 0:
        translation_task(args, args.input, args.output)
    else:
        # Split input into N files
        input_paths = [f'{args.working}/{i}.in' for i in range(args.jobs)]
        output_paths = [f'{args.working}/{i}.out' for i in range(args.jobs)]
        input_files = [open(path, mode='w', encoding='utf-8', errors='surrogateescape') for path in input_paths]
        with open(args.input, encoding='utf-8', errors='surrogateescape') as fin:
            for i, line in enumerate(fin):
                line = line.rstrip('\n')
                print(line, file=input_files[i % args.jobs])
        for f in input_files:
            f.close()

        # Submit jobs
        import submitit
        executor = submitit.AutoExecutor(folder=args.working)
        executor.update_parameters(
            name='m2m.{}2{}{}'.format(args.src, '2' + args.pivot if args.pivot else '', args.tgt),
            slurm_partition=args.partition,
            slurm_constraint='volta32gb',
            gpus_per_node=MODELS[args.model]['ngpus'],
            cpus_per_task=10,
            timeout_min=60*24*3,
        )
        if args.parallel_jobs is not None:
            executor.update_parameters(slurm_array_parallelism=args.parallel_jobs)
        jobs = executor.map_array(translation_task, [args for _ in range(args.jobs)], input_paths, output_paths)
        print(f'>>> submitted {len(jobs)} jobs')
        [job.result() for job in jobs]
        print('>>> done!')
        [os.remove(path) for path in input_paths]

        # Merge all the N output files
        output_files = [open(path, encoding='utf-8', errors='surrogateescape') for path in output_paths]
        with open(args.output, mode='w', encoding='utf-8') as fout:
            end = False
            while not end:
                for fin in output_files:
                    line = fin.readline()
                    if not line:
                        end = True
                        break
                    print(line, end='', file=fout)
        for f in output_files:
            f.close()
        [os.remove(path) for path in output_paths]

    # If the input/output are in jsonl format insert the translations into the original data
    if args.jsonl is not None:
        with open(args.output, encoding='utf-8') as fin:
            with open(original_output, mode='x', encoding='utf-8') as fout:
                for sample in jsonl_data:
                    for field in args.jsonl:
                        translation = fin.readline()
                        sample[field] = translation.strip()
                    print(json.dumps(sample), file=fout)


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    main()
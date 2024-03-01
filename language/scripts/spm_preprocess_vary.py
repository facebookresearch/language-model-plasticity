# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import argparse
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import sentencepiece as spm


def preprocess(spm_model_path, train_path, valid_path, test_path, dest_dir, 
               remove_empty=False, output_format='piece', workers=20, 
               ratio=1, subsample_train_only=False):
    print('Preprocessing Started!')
    base_ratio = ratio
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.dirname(spm_model_path) # Hack to maintain the train/valid/test tokenised files in the spm folder
        # Tokenize with SentencePiece
        for split, path in ('train', train_path), ('valid', valid_path), ('test', test_path):
            if subsample_train_only:
                ratio = base_ratio if split == 'train' else 1
            else:
                ratio = base_ratio
            if path is None:
                continue
            if path == '-':
                path = sys.stdin.fileno()
            
            #### Hack for count the size of the file
            # tot_lines = 0 
            # with open(path, encoding='utf-8', errors='surrogateescape') as fin:
            #     tot_lines = sum(1 for line in fin) 
            # tot_lines = 2104942434 # cc100-english
            # tot_lines = 417058636 # cc100-de
            tot_lines = 318479827 # cc100-th!!!!!!!!!!!!
            max_lines = int(tot_lines * ratio)
            print(ratio, max_lines)

            #### Tokenization
            with open(path, encoding='utf-8', errors='surrogateescape') as fin:
                with open(f'{tmp}/{split}', mode='w', encoding='utf-8', errors='surrogateescape') as fout:
                    encoder = MultiprocessingEncoder(model=spm_model_path, remove_empty=remove_empty, output_format=output_format)
                    pool = Pool(workers, initializer=encoder.initializer)
                    encoded_lines = pool.imap(encoder.encode, fin, 10000)
                    for i, line in enumerate(encoded_lines, start=1):
                        if line is not None:
                            print(line, file=fout)
                        if i % 10000 == 0:
                            print("tokenized {} lines".format(i), file=sys.stderr)
                        if i > max_lines:
                            print('Done tokenisation!')
                            pool.terminate()
                            break

            #### Duplicate fout for 1/ratio times
            if ratio < 0.16: # for thai
                copy_file = f'{tmp}/{split}' + 'copy'
                shutil.copyfile(f'{tmp}/{split}', copy_file)
                copy_times = int(0.16/ratio) - 1
                print(copy_times)
                print('wow')
                lang = spm_model_path.split('/spm/')[0].split('/')[-1] 
                with open(f'{tmp}/{split}', mode='a', encoding='utf-8', errors='surrogateescape') as fout:
                    for t in range(copy_times):
                        with open(copy_file, mode='r', encoding='utf-8', errors='surrogateescape') as copyin:
                            for line in copyin:
                                print(line, file=fout)
            print(f'{tmp}/{split}')
            #### Duplicate for low-resources languages -> no longer needed since we switch to 1 gpu training on 09/01/2023
            # lang = spm_model_path.split('/spm/')[0].split('/')[-1] 
            # if lang in ['test']:
            #     copy_file = f'{tmp}/{split}' + 'copy'
            #     shutil.copyfile(f'{tmp}/{split}', copy_file)
            #     copy_times = 1000
            #     print(copy_times)
            #     print('wow')
            #     with open(f'{tmp}/{split}', mode='a', encoding='utf-8', errors='surrogateescape') as fout:
            #         for t in range(copy_times):
            #             with open(copy_file, mode='r', encoding='utf-8', errors='surrogateescape') as copyin:
            #                 for line in copyin:
            #                     print(line, file=fout)

        # Generate dictionary
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        if output_format == 'piece':
            vocab = [sp.id_to_piece(i) for i in range(3, sp.vocab_size())]
        else:
            vocab = map(str, range(sp.vocab_size()))

        with open(f'{tmp}/dict.txt', mode='w', encoding='utf-8', errors='surrogateescape') as f:
            for word in vocab:
                print(word, 1, file=f)
        
        # Binarize
        command = [
            'fairseq-preprocess',
            # '--source-lang', src_lang, '--target-lang', tgt_lang,
            '--only-source',
            '--thresholdsrc', '0',
            '--destdir', dest_dir,
            '--srcdict', f'{tmp}/dict.txt',
            '--workers', str(workers),
        ]
        for split, path in ('train', train_path), ('valid', valid_path), ('test', test_path):
            if path is not None:
                command += [f'--{split}pref', f'{tmp}/{split}']
        subprocess.run(command)

        # Copy SentencePiece model
        shutil.copyfile(spm_model_path, f'{dest_dir}/sentencepiece.bpe.model')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='SentencePiece model')
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid')
    parser.add_argument('--test')
    parser.add_argument('--destdir', required=True)
    parser.add_argument('--remove-empty', action='store_true', help='remove empty lines')
    parser.add_argument('--output-format', choices=['id', 'piece'], default='piece')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--subsample-train-only', action='store_true', default=False)
    args = parser.parse_args()

    preprocess(
        spm_model_path=args.model,
        train_path=args.train, valid_path=args.valid, test_path=args.test, dest_dir=args.destdir,
        remove_empty=args.remove_empty, output_format=args.output_format, workers=args.workers,
        ratio=args.ratio, subsample_train_only=args.subsample_train_only
    )


class MultiprocessingEncoder(object):
    def __init__(self, model, remove_empty, output_format):
        self.model = model
        self.remove_empty = remove_empty
        self.output_format = output_format

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.model)

    def encode(self, line):
        global sp
        line = line.strip()
        if len(line) == 0 and self.remove_empty:
            return None

        if self.output_format == 'piece':
            return ' '.join(sp.encode_as_pieces(line))
        else:  # output_format == 'id'
            return ' '.join(map(str, sp.encode(line)))


if __name__ == "__main__":
    main()
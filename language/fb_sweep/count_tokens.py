# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count No Tokens')
    parser.add_argument('--lang', default='de')
    parser.add_argument('--bin_dir', default='bin')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    # data = "/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/wikipedia-20220601/{}/{}".format(args.lang, args.bin_dir)
    data = "/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/cc100/{}/{}".format(args.lang, args.bin_dir)
    dictionary = Dictionary.load(data + '/dict.txt')
    data = data + '/{}'.format(args.split) 

    dataset = load_indexed_dataset(data, dictionary)
    # print(len(dataset))
    # print(dataset.sizes)
    print('{}/{}/{}: {}'.format(args.lang, args.bin_dir, args.split, dataset.sizes.sum()))
    print('{}/{}/{}: {}'.format(args.lang, args.bin_dir, args.split, human_format(dataset.sizes.sum())))

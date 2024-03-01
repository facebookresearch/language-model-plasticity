# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import json
import random
MIN_WORDS=5
VALID_SIZE=1000


def main():
    parser = argparse.ArgumentParser(description='Process the multilingual wikipedia')
    parser.add_argument('--lang', type=str, choices=['en', 'de', 'fr', 'zh', 'zh_min_nan',
                                                     'fr','es','el','bg','ru','tr','ar',
                                                     'vi','th','hi','sw','ur',
                                                     'ay', 'gn', 'nah','qu'],
                        required=True)
    args = parser.parse_args()

    lang = args.lang
    DATA=f"/checkpoint/artetxe/storage/data/multilingual/wikipedia-20220601/jsonl/{lang}/{lang}wiki-20220601.train.jsonl"
    DESTDIR=f"/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/wikipedia-20220601/{lang}" #TODO: PUT BACK!

    print('Extracting data from json...')
    docs=[]
    with open(DATA, encoding="utf-8", errors="surrogateescape") as f:
        for i, line in enumerate(f):
            fields = json.loads(line)
            doc = fields.get('text', '').splitlines() # Each line is a document.
            doc = [line.strip() for line in doc if line.strip()]
            wordcount = sum([len(line.split()) for line in doc])
            if wordcount >= MIN_WORDS:
                docs.append(doc)

    print('Shuffling...')
    random.shuffle(docs)
    train = docs[VALID_SIZE * 2:]
    valid = docs[:VALID_SIZE]
    test = docs[VALID_SIZE: VALID_SIZE*2]

    print('Writing output files...')
    for data, split in (valid, 'valid'), (test, 'test'), (train, 'train'):
        print(split)
        with open(f"{DESTDIR}/wiki.{lang}.{split}.txt", mode="w", encoding="utf-8", errors="surrogateescape") as f:
            print('writing ...')
            for doc in data:
                print('\n'.join(doc), file=f)
                print(file=f)
    print('Done writing output files')


if __name__ == '__main__':
    main()


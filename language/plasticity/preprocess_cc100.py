# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import json
import random
VALID_SIZE=1000


def extract_documents(data):
    print('Extracting documents...')
    docs=[]
    with open(data, encoding="utf-8", errors="surrogateescape") as f:
        doc = []
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(i, line)
            line = line.strip()
            if line:
                doc.append(line)
            else: # line is empty, so it's the end of a document.
                #NOTE: English was preprocessed to exclude documents < 5 tokens.
                docs.append(doc)
                doc = []
    return docs


def extract_documents_with_minimum_words(data, min_words=5):
    print('Extracting documents...')
    docs=[]
    with open(data, encoding="utf-8", errors="surrogateescape") as f:
        doc = []
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(i, line)
            line = line.strip()
            if line:
                doc.append(line)
            else: # line is empty, so it's the end of a document.
                #NOTE: English was preprocessed to exclude documents < 5 tokens, otherwise the memory blows
                wordcount = sum([len(line.split()) for line in doc])
                if wordcount >= min_words:
                    docs.append(doc)
                doc = []
    return docs


def write(docs, destdir, split, valid_size=VALID_SIZE):
    print('Shuffling...')
    random.shuffle(docs) # this probably blows out the memory, check if there is any other implementations
    if split == 0:
        train = docs[valid_size * 2:]
        valid = docs[:valid_size]
        test = docs[valid_size: valid_size*2]
        print('Writing output files...')
        print(VALID_SIZE, len(docs), len(train), len(valid), len(test))
        for data, split in (valid, 'valid'), (test, 'test'), (train, 'train{}'.format(split)):
            with open(f"{destdir}/{split}.txt", mode="x", encoding="utf-8", errors="surrogateescape") as f:
                for doc in data:
                    print('\n'.join(doc), file=f)
                    print(file=f)
    else:
        train = docs
        print('Writing output files...')
        for data, split in (train, 'train{}'.format(split)),:
            with open(f"{destdir}/{split}.txt", mode="x", encoding="utf-8", errors="surrogateescape") as f:
                for doc in data:
                    print('\n'.join(doc), file=f)
                    print(file=f)
    print('Done')


def write_without_shuffle(docs, destdir, split):
    if split == 0:
        train = docs[VALID_SIZE * 2:]
        valid = docs[:VALID_SIZE]
        test = docs[VALID_SIZE: VALID_SIZE*2]
        print('Writing output files...')
        for data, split in (valid, 'valid'), (test, 'test'), (train, 'train{}'.format(split)):
            with open(f"{destdir}/{split}.txt", mode="x", encoding="utf-8", errors="surrogateescape") as f:
                for doc in data:
                    print('\n'.join(doc), file=f)
                    print(file=f)
    else:
        train = docs
        print('Writing output files...')
        for data, split in (train, 'train{}'.format(split)),:
            with open(f"{destdir}/{split}.txt", mode="x", encoding="utf-8", errors="surrogateescape") as f:
                for doc in data:
                    print('\n'.join(doc), file=f)
                    print(file=f)
    print('Done')


def main():
    parser = argparse.ArgumentParser(description='Preprocess the data for cc100')
    parser.add_argument('--lang', type=str, 
                        choices=['en', 'de', 'test', 'fr',
                                 'es', 'el', 'bg', 'ru', 'tr', 
                                 'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur', 
                                 'cc100_mini'],
                        required=True)
    parser.add_argument('--cc100_download_dir', type=str, default='/datasets01/cc100/031720')
    args = parser.parse_args()

    lang = args.lang
    cc100_download_dir = args.cc100_download_dir
    destdir=f"./plasticity/datasets/cc100/{lang}"
    languages = {'de': 'de_DE', 'fr': 'fr_XX', 'es': 'es_XX', 'el': 'el_GR', 'bg': 'bg_BG', # You might need to check the name of the text file for each language
                 'tr': 'tr_TR', 'ar': 'ar_AR', 'th': 'th_TH', 'zh': 'zh_CN', 'hi': 'hi_IN', 
                 'sw': 'sw_KE', 'ur': 'ur_PK', 
    }
    languages_special = {'en': 'en/raw/en.', 'ru': 'ru/raw/ru.', 'vi': 'vi/raw/vi.', 
                         'test': 'test/en.', 
                         'cc100_mini': 'cc100_small/cc100_mini.txt'
    }

    if lang == 'cc100_mini':
        data = f"./plasticity/datasets/cc100_small/cc100_mini.txt" 
        docs = extract_documents(data)
        write(docs, destdir, 0)
    elif lang in languages:
        data = f"{cc100_download_dir}/{languages[lang]}.txt" # CHANGE THE PATH TO YOUR CC100 DOWNLOAD
        docs = extract_documents(data) if lang not in ['en', 'cc100_mini'] else extract_documents_with_minimum_words(data)
        write(docs, destdir, 0 if lang != 'en' else None)
    elif lang == 'en':
        data_prefix="./plasticity/datasets/cc100/en/raw/en."
        for split in range(0, 10):
            docs = extract_documents_with_minimum_words(data_prefix + str(split))
            write(docs, destdir, split)
    elif lang == 'ru':
        data_prefix="./plasticity/datasets/cc100/ru/raw/ru."
        for split in range(0, 10):
            docs = extract_documents(data_prefix + str(split))
            write(docs, destdir, split)
    elif lang == 'vi':
        data_prefix="./plasticity/datasets/cc100/vi/raw/vi."
        for split in range(0, 10):
            docs = extract_documents(data_prefix + str(split))
            write(docs, destdir, split)
    elif lang == 'test':
        data_prefix="./plasticity/datasets/cc100/test/en."
        for split in range(0, 4):
            docs = extract_documents(data_prefix + str(split))
            write(docs, destdir, split)
    elif lang == 'cc100_mini':
        data = "./plasticity/datasets/cc100_small/cc100_mini.txt"
        docs = extract_documents(data)
        write_without_shuffle(docs, destdir, 0)     


if __name__ == '__main__':
    main()
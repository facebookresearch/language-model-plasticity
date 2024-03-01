# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import json
import os
import sentencepiece as spm
import string
import re

MLQA_LANGS = ['ar', 'de', 'es', 'en', 'hi', 'vi', 'zh']
XQUAD_LANGS = ['ar', 'de', 'el', 'es', 'hi', 'ru', 'th', 'tr', 'vi', 'zh']


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []

    tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
    return tokens

def is_whitespace(c):
    if (
        c == " " or c == "\t" or c == "\r" or c == "\n" or c == '\ufeff' or c == '\u2005' or c == '\u200f' or c == '˛' or c == '᾽' or
        ord(c) in {0x2009, 0x200B, 0x200C, 0x200D, 0x200E, 0x202F, 0x3000, 180, 160, 168, 730, 732, 175, 900, 65292, 65533}
    ):
        return True
    return False

def clean_text(text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def encode(output_type, sp, text):
    if output_type == 'pieces':
        return sp.EncodeAsPieces(text)
    else: # output_type = id
        return [str(i) for i in sp.encode(text)]

def process_file(args, input_file_name, split, lang):
    bad_qs = 0
    num_qs = 0
    sp = spm.SentencePieceProcessor()
    spm_file = args.spm_file_pattern.replace(args.spm_replacement_pattern, lang)
    sp.Load(spm_file)
    print('Loaded spm file', spm_file)

    s1_filename = os.path.join(args.output_folder, split + '.' + lang + '_1.txt')
    s2_filename = os.path.join(args.output_folder, split + '.' + lang + '_2.txt')
    s3_filename = os.path.join(args.output_folder, split + '.' + lang + '_3.txt')
    s4_filename = os.path.join(args.output_folder, split + '.' + lang + '_4.txt')
    id_filename = os.path.join(args.output_folder, split + '.' + lang + '.id')
    label_filename = os.path.join(args.output_folder, split + '.' + lang + '.lbl')

    skipped_answers_count = 0
    with open(input_file_name, 'r') as f_in, open(s1_filename, 'w') as s1_out, open(s2_filename, 'w') as s2_out, open(id_filename, 'w') as id_out, open(label_filename, 'w') as lbl_out, open(s3_filename, 'w') as s3_out, open(s4_filename, 'w') as s4_out:
        data = json.load(f_in)
        for example in data['data']:

            for p in example['paragraphs']:
                context = p['context']
                doc_tokens, char_to_word_offset = clean_text(context)

                orig_to_tok_index = []
                tok_to_orig_index = []
                all_doc_tokens = []
                word = -1
                bped_pieces = encode('pieces', sp, " ".join(doc_tokens))
                bped_out = encode(args.output_type, sp, " ".join(doc_tokens))
                assert len(bped_pieces) == len(bped_out)
                for x in doc_tokens:
                    tokens = encode('pieces', sp, x)
                    count = len([t for t in tokens if t.startswith('\u2581')])
                    if count > 1:
                        print(x)
                        import pdb; pdb.set_trace()

                for idx, b in enumerate(bped_pieces):
                    if b.startswith('\u2581') or idx == 0:
                        word += 1
                        orig_to_tok_index.append(len(all_doc_tokens))

                    all_doc_tokens.append(bped_out[idx])
                    tok_to_orig_index.append(word)

                assert (len(all_doc_tokens) == len(bped_pieces))
                try:
                    assert (word + 1 == len(doc_tokens))
                except AssertionError:
                    # import pdb; pdb.set_trace()
                    pass

                for qa in p['qas']:
                    cleaned_question, _ = clean_text(qa['question'])
                    q = " ".join(encode(args.output_type, sp,
                        " ".join(cleaned_question)))

                    if 'is_impossible' in qa and qa['is_impossible']:
                        continue
                    else:
                        answer = qa['answers'][0]
                        orig_answer_text = answer["text"]

                        if orig_answer_text == '':
                            # print("Empty answer")
                            continue
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            # import pdb; pdb.set_trace()
                            skipped_answers_count += 1
                            continue

                        tok_start_position = orig_to_tok_index[start_position]
                        if end_position < len(doc_tokens) - 1:
                            tok_end_position = orig_to_tok_index[end_position + 1] - 1
                        else:
                            tok_end_position = len(all_doc_tokens) - 1

                        (tok_start_position, tok_end_position) = _improve_answer_span(
                            args, all_doc_tokens, tok_start_position, tok_end_position,
                            sp, orig_answer_text)

                    num_qs += 1
                    print(' '.join(all_doc_tokens), file=s1_out) # why 4 files
                    print(q, file=s2_out)
                    print(' '.join(doc_tokens), file=s3_out)
                    print(' '.join([str(ii) for ii in tok_to_orig_index]), file=s4_out)
                    print(qa['id'], file=id_out)
                    # lbl_str = f'{1-int(is_impossible)}'

                    # if num_qs % 1000 == 0:
                    #     print(num_qs)
                    try:
                        assert (tok_end_position >= tok_start_position)
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    lbl_str = f'{tok_start_position} {tok_end_position}'
                    print(lbl_str, file=lbl_out)

    print('skipped answer count: ', skipped_answers_count)
    print('Has No Answer questions:', bad_qs, 'out of', num_qs)


def _improve_answer_span(args, doc_tokens, input_start, input_end, sp,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = sp.DecodePieces(encode('pieces', sp, orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = sp.DecodePieces(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text or text_span[1:] == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)


def main(args):
    process_file(args, args.train_file, 'train', 'en')
    process_file(args, args.squad_dev_file, 'squad_valid', 'en')

    for lang in XQUAD_LANGS:
        print(lang)
        train_fname = os.path.join(args.squad_translate_train_location, f"squad.translate.train.en-{lang}.json")
        process_file(args, train_fname, 'train', lang)

        dev_fname = os.path.join(args.squad_translate_dev_location, f"squad.translate.dev.en-{lang}.json")
        process_file(args, dev_fname, 'squad_valid', lang)

        test_fname = os.path.join(args.xquad_test_location, f"xquad.{lang}.json")
        process_file(args, test_fname, 'xquad_test', lang)

    for lang in MLQA_LANGS:
        print("MLQA-{0}".format(lang))
        dev_fname = os.path.join(
            args.mlqa_dev_test_location,
            'dev',
            'dev-context-{0}-question-{0}.json'.format(lang),
        )
        process_file(args, dev_fname, 'mlqa_valid', lang)

        test_fname = os.path.join(
            args.mlqa_dev_test_location,
            'test',
            'test-context-{0}-question-{0}.json'.format(lang),
        )
        process_file(args, test_fname, 'mlqa_test', lang)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-impossible', default=False, action='store_true')
    parser.add_argument('--spm-replacement-pattern',
        default='YOUR_LANGUAGE_HERE', help="""We pass a string pattern to 
                                              --spm-file-pattern. This parameter specifies which part of the 
                                              string to replace with the language abbreiation""") 
    parser.add_argument('--spm-file-pattern', default='plasticity/datasets/cc100/YOUR_LANGUAGE_HERE/spm/spm.bpe.model')
    parser.add_argument('--output-type', choices=['id', 'pieces'], default='id')
    parser.add_argument('--train-file', default='plasticity/datasets/squad/train-v1.1.json')
    parser.add_argument('--squad-dev-file', default='plasticity/datasets/squad/dev-v1.1.json')
    parser.add_argument('--mlqa-dev-test-location', default='plasticity/datasets/MLQA/')
    parser.add_argument('--squad-translate-train-location', default='plasticity/datasets/squad/translate-train/')
    parser.add_argument('--squad-translate-dev-location', default='plasticity/datasets/squad/translate-dev/')
    parser.add_argument('--xquad-test-location', default='plasticity/datasets/XQUAD/xquad/')
    parser.add_argument('--output-folder', default='plasticity/datasets/XQUAD_MLQA/processed')
    args = parser.parse_args()
    main(args)
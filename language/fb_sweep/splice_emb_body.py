# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

"""
    Splice the separately trained embedding and body together
    Yihong 26-07-2022
"""
import copy
import torch
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True)
    parser.add_argument('--body_path', required=True)
    parser.add_argument('--splice_path', required=True)
    parser.add_argument('--new_data_path', default=None)
    parser.add_argument('--new_vocab_size', default=None, type=int)
    
    args = parser.parse_args()

    emb_path = args.emb_path
    body_path = args.body_path
    splice_path = args.splice_path

    emb_model = torch.load(emb_path)
    body_model = torch.load(body_path)
    
    # splice them
    new_model = copy.deepcopy(body_model)
    new_model['model']['encoder.sentence_encoder.embed_tokens.weight'] = emb_model['model']['encoder.sentence_encoder.embed_tokens.weight']
    new_model['model']['encoder.lm_head.weight'] = emb_model['model']['encoder.lm_head.weight']
    if args.new_data_path != None:
        new_model['cfg']['task']['data'] = args.new_data_path
    if args.new_vocab_size != None:
        if args.new_vocab_size > 50005:
            new_model['model']['encoder.lm_head.bias'] = new_model['model']['encoder.lm_head.bias'].repeat((args.new_vocab_size // 50005)+1)[:args.new_vocab_size]
        else:
            new_model['model']['encoder.lm_head.bias'] = new_model['model']['encoder.lm_head.bias'][:args.new_vocab_size]
    
    print('Save New Model {}'.format(splice_path))
    folder_path = Path(splice_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = str(folder_path) + '/checkpoint_best.pt'
    torch.save(new_model, model_path)

    if 'nli' or 'image_speaks' or 'senti' in model_path:
        for split in 'input0', 'input1', 'label':
            new_dict_path = Path(str(folder_path) + '/' + split + '/dict.txt')
            new_dict_path.parent.mkdir(parents=True, exist_ok=True)
            if split in ['input0', 'input1']:
                dict_path = emb_model['cfg']['task']['data'] + '/dict.txt' # language adaptation dict
            else:
                dict_path = body_model['cfg']['task']['data'] + '/label/dict.txt' # mnli body label dict
            shutil.copyfile(dict_path, new_dict_path)
    print('Done saving the new model')
    

if __name__ == '__main__':
    main()
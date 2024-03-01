# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

"""
    Load a pretrained model, manually reset emb
"""
import ast
import copy
import torch
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True)
    parser.add_argument('--reset_path', required=True)
    parser.add_argument('--keep_gaussian', default=False, type=ast.literal_eval)
    
    args = parser.parse_args()

    emb_path = args.emb_path
    reset_path = args.reset_path
    keep_gaussian = args.keep_gaussian

    emb_model = torch.load(emb_path)

    mean = emb_model['model']['encoder.sentence_encoder.embed_tokens.weight'].mean()
    std = emb_model['model']['encoder.sentence_encoder.embed_tokens.weight'].std()
    
    # reset the emb to init states
    new_model = copy.deepcopy(emb_model)
    torch.manual_seed(0)
    if keep_gaussian:
        torch.nn.init.normal_(new_model['model']['encoder.sentence_encoder.embed_tokens.weight'], mean, std)
    else:
        torch.nn.init.normal_(new_model['model']['encoder.sentence_encoder.embed_tokens.weight'], mean=0, std=0.02)
    padding_idx = new_model['cfg']['model'].pad
    new_model['model']['encoder.sentence_encoder.embed_tokens.weight'][padding_idx].fill_(0)
    new_model['model']['encoder.lm_head.weight'] = copy.deepcopy(new_model['model']['encoder.sentence_encoder.embed_tokens.weight'])
    
    print('Save New Model {}'.format(reset_path))
    folder_path = Path(reset_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = str(folder_path) + '/checkpoint_best.pt'
    torch.save(new_model, model_path)
    print('Done saving the new model')
    

if __name__ == '__main__':
    main()
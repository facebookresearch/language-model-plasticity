# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

LABEL_FN = lambda x: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}[x]
# arabic #LABEL_FN = lambda x: {2: 'contradiction', 1: 'neutral', 0: 'entailment'}[x]
BASEDIR = "/checkpoint/yhc/inductivise-lm/inductivise-lm/"
EN_SPM = '/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/cc100/en/spm/cc100.en.spm.model'
AR_SPM = '/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/cc100/ar/spm/cc100.ar.spm.model'
MODELS = {
    "standard_NA_train-all-en": {
        "model_path": BASEDIR + "exps/nli-20220722-1223/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.wu7363.mu122720.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": LABEL_FN,
    },    
    "standard_NA_finetune-all-en": {
        "model_path": BASEDIR + "exps/cc100/nli-full-roberta-base/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu122720.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": LABEL_FN,
    },
    "forget_NA_finetune-all-en": { 
        "model_path": BASEDIR + "exps/cc100/nli-full-clip0.5.adamef.k1000/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu122720.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": LABEL_FN,
    },
    "standard_standard-ar": {
        "model_path": BASEDIR + "exps/cc100/nli-full-roberta-base-ar/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu2090.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": AR_SPM},
        "label_fn": LABEL_FN,
    },
    "forget_standard-ar": {
        "model_path": BASEDIR + "exps/cc100/nli-full-clip0.5.adamef.k1000-ar/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu2090.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": AR_SPM},
        "label_fn": LABEL_FN,
    },
    "standard_adapt-emb-zh_finetune-body-en": {
        "model_path": "",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": None,
    },
    "standard_shallow-adapt-emb-zh_finetune-body-en": {
        "model_path": "",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": None,
    },
    "forget_adapt-emb-zh_finetune-body-en": {
        "model_path": "",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": EN_SPM},
        "label_fn": None,
    },
}
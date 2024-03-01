# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

LABEL_FN = lambda x: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}[x]
BASEDIR = "/checkpoint/yhc/inductivise-lm/inductivise-lm/"
TH_SPM = BASEDIR + "datasets/cc100/cc100_mini/spm/cc100.cc100_mini.spm.model" 

MODELS = {
    # TH_SPM = BASEDIR + "datasets/cc100/th/spm/cc100.th.spm.model" 
    "forget_adapt-emb-th5000000.0-step125000_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/multilingual_pretrain_nli/forget_adapt-emb-th5000000.0-step125000_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th5000000.0-step125000_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/multilingual_pretrain_nli/standard_adapt-emb-th5000000.0-step125000_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },


    "forget_NA_finetune-all-en": {
         "model_path": BASEDIR + "exps/cc100_mini/nli-multilingual_full-clip0.5.adamef.k1000/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu122720.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_NA_finetune-all-en": {
         "model_path": BASEDIR + "exps/cc100_mini/nli-multilingual_full-roberta-base/NLI.fp16.sentpred.bos0.sep2.roberta_base.adam.b2_0.98.eps1e-06.clip0.0.lr1e-05.mu122720.dr0.1.atdr0.1.wd0.01.ms32.uf1.s1.ngpu1/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },
}

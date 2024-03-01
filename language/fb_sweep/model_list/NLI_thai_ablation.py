# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

LABEL_FN = lambda x: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}[x]
BASEDIR = "/checkpoint/yhc/inductivise-lm/inductivise-lm/"
TH_SPM = BASEDIR + "datasets/cc100/th/spm/cc100.th.spm.model" 

MODELS = {

    "standard_adapt-emb-th5000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th5000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th5000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th5000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th5000000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th5000000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th5000000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th5000000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th1000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th1000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th1000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th1000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th1000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th1000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th1000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th1000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th1000000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th1000000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th1000000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th1000000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th10000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th10000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th10000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th10000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th10000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th10000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th10000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th10000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th100000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th100000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th100000000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th100000000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "forget_adapt-emb-th100000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/forget_adapt-emb-th100000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },

    "standard_adapt-emb-th100000.0-stepbest_finetune-body-en": {
         "model_path": BASEDIR + "exps/splice/nli/standard_adapt-emb-th100000.0-stepbest_finetune-body-en/checkpoint_best.pt",
        "model_overrides": {"bpe": "sentencepiece", "sentencepiece_model": TH_SPM},
        "label_fn": LABEL_FN,
    },
}

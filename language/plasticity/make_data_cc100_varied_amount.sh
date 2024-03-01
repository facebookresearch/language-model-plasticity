# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

preprocess() {
    lang=$1
    ratio=$2 # how much percentage of data to binarize

    DATADIR=plasticity/datasets/cc100/$lang
    SPM_DIR=$DATADIR/spm

    BINDIR=$DATADIR/bin-fixed-valid-dupli-$ratio # keep the same validation set
    mkdir -p $BINDIR
    
    ######## Step 3 Binarize ############### only subsample the training set but keep the same validation set
    python scripts/spm_preprocess.py \
        --model $SPM_DIR/spm.bpe.model \
        --train $DATADIR/train.txt \
        --valid $DATADIR/valid.txt \
        --destdir $BINDIR \
        --output-format id \
        --ratio $ratio \
        --subsample-train-only \
        &> $BINDIR/log
    echo "done ${lang} ${ratio}"
}


preprocess_vary() {
    lang=$1
    ratio=$2 # how much percentage of data to binarize
    bindir_name=$3

    DATADIR=plasticity/datasets/cc100/$lang
    SPM_DIR=$DATADIR/spm

    BINDIR=$DATADIR/bin-fixed-valid-dupli-$bindir_name # keep the same validation set
    mkdir -p $BINDIR
    
    ######## Step 3 Binarize ############### only subsample the training set but keep the same validation set
    python scripts/spm_preprocess_vary.py \
        --model $SPM_DIR/cc100.$lang.spm.model \
        --train $DATADIR/train.txt \
        --valid $DATADIR/valid.txt \
        --destdir $BINDIR \
        --output-format piece \
        --ratio $ratio \
        --subsample-train-only \
        &> $BINDIR/log
    echo "done ${lang} ${ratio}"
}


preprocess_5M() {
    lang=$1
    ratio=$2 # how much percentage of data to binarize

    DATADIR=plasticity/datasets/cc100/$lang
    SPM_DIR=$DATADIR/spm

    BINDIR=$DATADIR/bin-fixed-valid-dupli-5M # keep the same validation set
    mkdir -p $BINDIR
    
    ######## Step 3 Binarize ############### only subsample the training set but keep the same validation set
    python scripts/spm_preprocess.py \
        --model $SPM_DIR/spm.bpe.model \
        --train $DATADIR/train.txt \
        --valid $DATADIR/valid.txt \
        --destdir $BINDIR \
        --output-format id \
        --ratio $ratio \
        --subsample-train-only \
        &> $BINDIR/log
    echo "done ${lang} ${ratio}"
}

# python fb_sweep/count_tokens.py --lang ar --bin_dir bin;
# python fb_sweep/count_tokens.py --lang bg --bin_dir bin; 
# python fb_sweep/count_tokens.py --lang de --bin_dir bin;
# python fb_sweep/count_tokens.py --lang el --bin_dir bin;
# python fb_sweep/count_tokens.py --lang en --bin_dir bin;
# python fb_sweep/count_tokens.py --lang es --bin_dir bin;
# python fb_sweep/count_tokens.py --lang fr --bin_dir bin;
# python fb_sweep/count_tokens.py --lang hi --bin_dir bin;
# python fb_sweep/count_tokens.py --lang ru --bin_dir bin;
# python fb_sweep/count_tokens.py --lang sw --bin_dir bin;
# python fb_sweep/count_tokens.py --lang th --bin_dir bin;
# python fb_sweep/count_tokens.py --lang tr --bin_dir bin;
# python fb_sweep/count_tokens.py --lang ur --bin_dir bin;
# python fb_sweep/count_tokens.py --lang vi --bin_dir bin;
# python fb_sweep/count_tokens.py --lang zh --bin_dir bin;


# 5M tokens
# preprocess_5M sw 0.014482860234434696;
# preprocess_5M ur 0.0060105864988936625;
# preprocess_5M hi 0.0023503710109293433;
# preprocess_5M ar 0.0012044704740684974;
# preprocess_5M tr 0.0011942492751921415;
# preprocess_5M th 0.0008213668057066923;
# preprocess_5M el 0.0008190930851074491;
# preprocess_5M bg 0.0006330481289578667;
# preprocess_5M zh 0.000514644908858894;
# preprocess_5M es 0.00042958866861610384;
# preprocess_5M fr 0.00037644153166829753;
# preprocess_5M de 0.00035512567696529103;
# preprocess_5M vi 0.0001729319606221278;
# preprocess_5M ru 0.00014319390398944277;



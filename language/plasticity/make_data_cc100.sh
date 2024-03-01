# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

preprocess() {
    lang=$1
    size=$2

    python plasticity/preprocess_cc100.py --lang $lang --cc100_download_dir $CC100_DOWNLOAD_DIR

    DATADIR=plasticity/datasets/cc100/$lang
    SPM_DIR=$DATADIR/spm
    mkdir -p $SPM_DIR

    cat $DATADIR/train* > $DATADIR/train.txt

    spm_train --input $DATADIR/train.txt \
    	--model_prefix $SPM_DIR/cc100.$lang.spm \
    	--vocab_size=$size \
    	--character_coverage=0.9998 \
        --input_sentence_size=5000000 --shuffle_input_sentence=true \
        &> $SPM_DIR/cc100.$lang.spm.log

    BINDIR=$DATADIR/bin
    mkdir -p $BINDIR
    python scripts/spm_preprocess.py \
        --model $SPM_DIR/cc100.$lang.spm.model \
        --train $DATADIR/train.txt \
        --valid $DATADIR/valid.txt \
        --destdir $BINDIR \
        --output-format piece \
        &> $BINDIR/log
}


#!/bin/bash
preprocess_cc100_mini_correct() {
    lang=$1
    size=$2

    #python inductivise-lm/preprocess_cc100.py --lang $lang

    DATADIR=/checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/cc100/$lang
    SPM_DIR=$DATADIR/spm
    #mkdir -p $SPM_DIR

    #cat $DATADIR/train* > $DATADIR/train.txt

    # spm_train --input $DATADIR/train.txt \
    # 	--model_prefix $SPM_DIR/cc100.$lang.spm \
    # 	--vocab_size=$size \
    # 	--character_coverage=0.9998 \
    #     --input_sentence_size=5000000 --shuffle_input_sentence=true \
    #     &> $SPM_DIR/cc100.$lang.spm.log

    BINDIR=$DATADIR/bin_correct
    mkdir -p $BINDIR
    python3 scripts/spm_preprocess.py \
        --model $SPM_DIR/cc100.$lang.spm.model \
        --train $DATADIR/train.txt \
        --valid $DATADIR/valid.txt \
        --destdir $BINDIR \
        --output-format piece \
        &> $BINDIR/log
}
# Note: If you have OOM issue when preprocess the CC100, Split Large File into 10 Chunks, 
## Shuffle within each chunk instead of globally to avoid the OOM issue
export CC100_DOWNLOAD_DIR='SOMEWHERE_TO_PUT_LARGE_DATASET'
export WORK_DIR='./plasticity/'
EXP_DIR="${WORK_DIR}exps/"
DAT_DIR="${WORK_DIR}datasets/"
mkdir -p $DAT_DIR/cc100/en/raw
mkdir -p $DAT_DIR/cc100/ru/raw
mkdir -p $DAT_DIR/cc100/vi/raw
cd $DAT_DIR/cc100/en/raw
split --verbose -nl/10 /datasets01/cc100/031720/en_XX.txt en.
mv en.aa en.0
mv en.ab en.1
mv en.ac en.2
mv en.ad en.3
mv en.ae en.4
mv en.af en.5
mv en.ag en.6
mv en.ah en.7
mv en.ai en.8
mv en.aj en.9
cd -
##
# cd $DAT_DIR/cc100/ru/raw
# split --verbose -nl/10 /datasets01/cc100/031720/ru_RU.txt ru.
# mv ru.aa ru.0
# mv ru.ab ru.1
# mv ru.ac ru.2
# mv ru.ad ru.3
# mv ru.ae ru.4
# mv ru.af ru.5
# mv ru.ag ru.6
# mv ru.ah ru.7
# mv ru.ai ru.8
# mv ru.aj ru.9
# cd -
#
# cd /checkpoint/yhc/inductivise-lm/inductivise-lm/datasets/vi/raw
# split --verbose -nl/10 /datasets01/cc100/031720/vi_VN.txt vi.
# mv vi.aa vi.0
# mv vi.ab vi.1
# mv vi.ac vi.2
# mv vi.ad vi.3
# mv vi.ae vi.4
# mv vi.af vi.5
# mv vi.ag vi.6
# mv vi.ah vi.7
# mv vi.ai vi.8
# mv vi.aj vi.9
# cd -
##

############ Preprocess for each language ##########
preprocess en 50000
# preprocess ru 50000
# preprocess vi 50000
# preprocess de 50000
# preprocess th 50000
# preprocess hi 50000
# preprocess ar 50000


###########
# preprocess_cc100_mini_correct cc100_mini 50000
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

DATA_DIR="./plasticity/datasets/XQUAD_MLQA/processed"
OUT_DIR="./plasticity/datasets/XQUAD_MLQA/"
OUT_BIN="./plasticity/datasets/XQUAD_MLQA/bin/"
mkdir -p $OUT_BIN

REPLACEMENT_STR="YOUR_LANGUAGE_HERE"
DICT_STR="./plasticity/datasets/cc100/YOUR_LANGUAGE_HERE/bin/dict.txt"


for LANG in ar de el en es hi ru th tr vi zh; do
    FAIRSEQ_DICT="${DICT_STR/"$REPLACEMENT_STR"/"$LANG"}"
    echo Using dictionary: $FAIRSEQ_DICT

    for FILE_NUM in 1 2; do
        python -m fairseq_cli.preprocess \
            --only-source \
            --srcdict ${FAIRSEQ_DICT} \
            --trainpref "${DATA_DIR}/train.${LANG}_${FILE_NUM}.txt" \
            --destdir "${OUT_DIR}/${LANG}/bin/";

        mv "${OUT_DIR}/${LANG}/bin/train.bin" "${OUT_BIN}/train_${LANG}_${FILE_NUM}.bin"
        mv "${OUT_DIR}/${LANG}/bin/train.idx" "${OUT_BIN}/train_${LANG}_${FILE_NUM}.idx"

        python -m fairseq_cli.preprocess \
            --only-source \
            --srcdict ${FAIRSEQ_DICT} \
            --trainpref "${DATA_DIR}/squad_valid.${LANG}_${FILE_NUM}.txt" \
            --destdir "${OUT_DIR}/${LANG}/bin/";

        mv "${OUT_DIR}/${LANG}/bin/train.bin" "${OUT_BIN}/squad_valid_${LANG}_${FILE_NUM}.bin"
        mv "${OUT_DIR}/${LANG}/bin/train.idx" "${OUT_BIN}/squad_valid_${LANG}_${FILE_NUM}.idx"

        if [ $LANG != 'en' ]; then
           python -m fairseq_cli.preprocess \
               --only-source \
               --srcdict ${FAIRSEQ_DICT} \
               --trainpref "${DATA_DIR}/xquad_test.${LANG}_${FILE_NUM}.txt" \
               --destdir "${OUT_DIR}/${LANG}/bin/";

           mv "${OUT_DIR}/${LANG}/bin/train.bin" "${OUT_BIN}/xquad_test_${LANG}_${FILE_NUM}.bin"
           mv "${OUT_DIR}/${LANG}/bin/train.idx" "${OUT_BIN}/xquad_test_${LANG}_${FILE_NUM}.idx"
        fi
    done
    for FILE_NUM in 3 4; do
        cp "${DATA_DIR}/train.${LANG}_${FILE_NUM}.txt" "${OUT_BIN}/train_${LANG}_${FILE_NUM}.txt"
        cp "${DATA_DIR}/squad_valid.${LANG}_${FILE_NUM}.txt" "${OUT_BIN}/squad_valid_${LANG}_${FILE_NUM}.txt"
        if [ $LANG != 'en' ]; then
           cp "${DATA_DIR}/xquad_test.${LANG}_${FILE_NUM}.txt" "${OUT_BIN}/xquad_test_${LANG}_${FILE_NUM}.txt"
        fi
    done
    for FILE_EXT in id lbl; do
        cp "${DATA_DIR}/train.${LANG}.${FILE_EXT}" "${OUT_BIN}/train_${LANG}.${FILE_EXT}"
        cp "${DATA_DIR}/squad_valid.${LANG}.${FILE_EXT}" "${OUT_BIN}/squad_valid_${LANG}.${FILE_EXT}"
        if [ $LANG != 'en' ]; then
           cp "${DATA_DIR}/xquad_test.${LANG}.${FILE_EXT}" "${OUT_BIN}/xquad_test_${LANG}.${FILE_EXT}"
        fi
    done
done

for LANG in ar de es en hi vi zh; do
    FAIRSEQ_DICT="${DICT_STR/"$REPLACEMENT_STR"/"$LANG"}"
    echo Using dictionary: $FAIRSEQ_DICT

    for FILE_NUM in 1 2
    do
        python -m fairseq_cli.preprocess \
            --only-source \
            --srcdict ${FAIRSEQ_DICT} \
            --trainpref "${DATA_DIR}/mlqa_valid.${LANG}_${FILE_NUM}.txt" \
            --destdir "${OUT_DIR}/${LANG}/bin/";

        mv "${OUT_DIR}/${LANG}/bin/train.bin" "${OUT_BIN}/mlqa_valid_${LANG}_${FILE_NUM}.bin"
        mv "${OUT_DIR}/${LANG}/bin/train.idx" "${OUT_BIN}/mlqa_valid_${LANG}_${FILE_NUM}.idx"

        python -m fairseq_cli.preprocess \
            --only-source \
            --srcdict ${FAIRSEQ_DICT} \
            --trainpref "${DATA_DIR}/mlqa_test.${LANG}_${FILE_NUM}.txt" \
            --destdir "${OUT_DIR}/${LANG}/bin/";

        mv "${OUT_DIR}/${LANG}/bin/train.bin" "${OUT_BIN}/mlqa_test_${LANG}_${FILE_NUM}.bin"
        mv "${OUT_DIR}/${LANG}/bin/train.idx" "${OUT_BIN}/mlqa_test_${LANG}_${FILE_NUM}.idx"
    done
    for FILE_NUM in 3 4; do
        cp "${DATA_DIR}/mlqa_valid.${LANG}_${FILE_NUM}.txt" "${OUT_BIN}/mlqa_valid_${LANG}_${FILE_NUM}.txt"
        cp "${DATA_DIR}/mlqa_test.${LANG}_${FILE_NUM}.txt" "${OUT_BIN}/mlqa_test_${LANG}_${FILE_NUM}.txt"
    done
    for FILE_EXT in id lbl; do
        cp "${DATA_DIR}/train.${LANG}.${FILE_EXT}" "${OUT_DIR}/${LANG}/bin/train_${LANG}.${FILE_EXT}"
        cp "${DATA_DIR}/mlqa_valid.${LANG}.${FILE_EXT}" "${OUT_BIN}/mlqa_valid_${LANG}.${FILE_EXT}"
        cp "${DATA_DIR}/mlqa_test.${LANG}.${FILE_EXT}" "${OUT_BIN}/mlqa_test_${LANG}.${FILE_EXT}"
    done
done
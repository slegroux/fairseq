#!/usr/bin/env bash
# 10/2021 (c) Sylvain Le Groux <syl20@cisco.com>

stage=$1

LS_ROOT=/home/syl20/data/en/librispeech/ 
SAVE_DIR=/home/syl20/data/en/Models/fairseq
CHECKPOINT_FILENAME=librispeech_transformer_l.pt
SUBSET=test-clean.tiny
CONFIG=$DATA/en/librispeech/config.yaml

if [ $stage -eq 0 ]; then
    python examples/speech_to_text/prep_librispeech_data.py \
        --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
fi

if [ $stage -eq 1 ]; then
    fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
        --config-yaml ${CONFIG} --train-subset train-clean-100,train-clean-360,train-other-500 --valid-subset dev-clean,dev-other \
        --num-workers 4 --max-tokens 40000 --max-update 300000 \
        --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
        --arch s2t_transformer_s --share-decoder-input-output-embed \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --clip-norm 10.0 --seed 1 --update-freq 8
fi


if [ $stage -eq 2 ]; then
    CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
    python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
        --num-epoch-checkpoints 10 \
        --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
fi



if [ $stage -eq 3 ]; then
    fairseq-generate ${LS_ROOT} --config-yaml ${CONFIG} --gen-subset ${SUBSET} \
        --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
        --max-tokens 50000 --beam 5 --scoring wer
fi

# fairseq-interactive ${LS_ROOT} --config-yaml config.yaml --task speech_to_text \
#   --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5
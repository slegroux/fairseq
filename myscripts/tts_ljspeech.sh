#!/usr/bin/env bash
# 10/2021 (c) Sylvain Le Groux <syl20@cisco.com>

AUDIO_DATA_ROOT=$DATA/en/LJSpeech-1.1
AUDIO_MANIFEST_ROOT=$AUDIO_DATA_ROOT
FEATURE_MANIFEST_ROOT=$AUDIO_DATA_ROOT
SAVE_DIR=examples/speech_synthesis/checkpoints
# CONFIG=examples/speech_synthesis/config.yaml

CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt

stage=$1

if [ $stage -eq 0 ]; then
	python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
		--output-data-root ${AUDIO_DATA_ROOT} \
		--output-manifest-root ${AUDIO_MANIFEST_ROOT}
fi

if [ $stage -eq 1 ]; then
	python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
	--audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
	--output-root ${FEATURE_MANIFEST_ROOT} \
	--ipa-vocab --use-g2p
	
fi

if [ $stage -eq 2 ]; then
	fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
		--tensorboard-logdir ${SAVE_DIR}/logdir \
  		--config-yaml config.yaml --train-subset train --valid-subset dev \
		--num-workers 4 --max-tokens 30000 --max-update 200000 \
		--task text_to_speech --criterion tacotron2 --arch tts_transformer \
		--clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
		--dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
		--encoder-normalize-before --decoder-normalize-before \
		--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--seed 1 --eval-inference --best-checkpoint-metric mcd_loss
fi

if [ $stage -eq 3 ]; then
	echo "START stage 3"
	python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
	--num-epoch-checkpoints 5 \
	--output ${CHECKPOINT_PATH}
	echo "STOP stage 3"
fi

DATA_DIR=/home/syl20/data/en/LJSpeech-1.1
FEATURE_MANIFEST_ROOT=${DATA_DIR}

SPLIT=${DATA_DIR}/test.tiny
SAVE_DIR=$DATA/en/Models/fairseq/ljspeech_fastspeech2_phn

CONFIG=${SAVE_DIR}/config.yaml
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_avg_last_5.pt


if [ $stage -eq 4 ]; then
	python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
	--config-yaml ${CONFIG} --gen-subset ${SPLIT} --task text_to_speech \
	--path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
	--dump-waveforms
fi
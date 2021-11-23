#!/usr/bin/env bash
# 10/2021 (c) Sylvain Le Groux <syl20@cisco.com>

stage=$1

AUDIO_DATA_ROOT=${DATA}/en/vctk_old
AUDIO_MANIFEST_ROOT=${AUDIO_DATA_ROOT}/fairseq/manifests
FEATURE_MANIFEST_ROOT=${AUDIO_DATA_ROOT}/fairseq/features
CONFIG=${FEATURE_MANIFEST_ROOT}/config.yaml
PROCESSED_DATA_ROOT=${AUDIO_DATA_ROOT}/fairseq/data
SAVE_DIR=${AUDIO_DATA_ROOT}/fairseq/checkpoints
LOG_DIR=${SAVE_DIR}/logdir
WANDB_PROJECT=fairseq

# preprocessing
if [ $stage -eq 0 ]; then
	python -m examples.speech_synthesis.preprocessing.get_vctk_audio_manifest \
	--output-data-root ${AUDIO_DATA_ROOT} \
	--output-manifest-root ${AUDIO_MANIFEST_ROOT}
fi

# feature comp
if [ $stage -eq 1 ]; then
	python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
	--audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
	--output-root ${FEATURE_MANIFEST_ROOT} \
	--ipa-vocab --use-g2p
fi

# denoise
if [ $stage -eq 2 ]; then
	for SPLIT in dev test train; do
		python -m examples.speech_synthesis.preprocessing.denoise_and_vad_audio \
		--audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
		--output-dir ${PROCESSED_DATA_ROOT} \
		--denoise --vad --vad-agg-level 3
	done
fi

# train

if [ $stage -eq 3 ]; then
	fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
	--config-yaml ${CONFIG} --train-subset train --valid-subset dev \
	--num-workers 4 --max-tokens 30000 --max-update 200000 \
	--task text_to_speech --criterion tacotron2 --arch tts_transformer \
	--clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
	--dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
	--encoder-normalize-before --decoder-normalize-before \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss \
	--tensorboard-logdir ${LOG_DIR} --wandb-project ${WANDB_PROJECT} \
	# --amp --reset-optimizer
fi

# inference

CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
if [ $stage -eq 4 ]; then
	python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
	--num-epoch-checkpoints 5 \
	--output ${CHECKPOINT_PATH}
fi

SPLIT=test.10
RESULTS_PATH=${AUDIO_DATA_ROOT}/fairseq/results
if [ $stage -eq 5 ]; then
	python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
	--config-yaml ${CONFIG} --gen-subset ${SPLIT} --task text_to_speech \
	--vocoder 'griffin_lim' \
	--path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
	--dump-waveforms --dump-features --dump-plots --dump-target \
	--results-path ${RESULTS_PATH}
fi
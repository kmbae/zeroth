#!/bin/bash
#
# Based mostly on the WSJ/Librispeech recipe. The training database is #####,
# it consists of 51hrs korean speech with cleaned automatic transcripts:
#
# http://www.openslr.org/resources (Mirror).
#
# Copyright  2017  Atlas Guide (Author : Lucas Jo)
#            2017  Gridspace Inc. (Author: Wonkyum Lee)
#
# Apache 2.0
#

# Check list before start
# 1. locale setup
# 2. pre-installed package: awscli, flac, sox, same cuda library, unzip
# 3. pre-install or symbolic link for easy going: rirs_noises.zip (takes pretty long time)
# 4. parameters: nCPU, num_jobs_initial, num_jobs_final, --max-noises-per-minute

data=./speechDATA
nCPU=30

. ./cmd.sh
. ./path.sh

# you might not want to do this for interactive shells.
set -e

startTime=$(date +'%F-%H-%M')
echo "started at" $startTime

# download the audio data and LMs
local/download_from_openslr.sh

# format the data as Kaldi data directories
for part in train_data_01 test_data_01; do
	# use underscore-separated names in data directories.
	local/data_prep.sh $data/$part data/$(echo $part | sed s/-/_/g)
done

# update segmentation of transcripts
for part in train_data_01 test_data_01; do
	local/updateSegmentation.sh data/$part data/local/lm
done

# prepare dictionary and language model
local/prepare_dict.sh data/local/lm data/local/dict_nosp

utils/prepare_lang.sh data/local/dict_nosp \
	"<UNK>" data/local/lang_tmp_nosp data/lang_nosp

local/format_lms.sh --src-dir data/lang_nosp data/local/lm

# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
# it takes long time and do this again after computing silence prob.
# you can do comment out here this time

#utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.tg.arpa.gz \
#	data/lang_nosp data/lang_nosp_test_tglarge
#utils/build_const_arpa_lm.sh data/local/lm/zeroth.lm.fg.arpa.gz \
#	  data/lang_nosp data/lang_nosp_test_fglarge

# Feature extraction (MFCC)
mfccdir=mfcc
hostInAtlas="ares hephaestus jupiter neptune"
if [[ ! -z $(echo $hostInAtlas | grep -o $(hostname -f)) ]]; then
  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  utils/create_split_dir.pl /mnt/{ares,hephaestus,jupiter,neptune}/$USER/kaldi-data/zeroth-kaldi/s5/$mfcc/storage \
    $mfccdir/storage
fi
for part in train_data_01 test_data_01; do
	steps/make_mfcc.sh --cmd "$train_cmd" --nj $nCPU data/$part exp/make_mfcc/$part $mfccdir
	steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

# ... and then combine data sets into one (for later extension)
utils/combine_data.sh \
  data/train_clean data/train_data_01

utils/combine_data.sh \
  data/test_clean data/test_data_01

# Make some small data subsets for early system-build stages.
#utils/subset_data_dir.sh --shortest data/train_clean 2000 data/train_2kshort
#utils/subset_data_dir.sh data/train_clean 5000 data/train_5k
#utils/subset_data_dir.sh data/train_clean 10000 data/train_10k


finishTime=$(date +'%F-%H-%M')
echo "GMM trainig is finished at" $finishTime
#exit
## online chain recipe using only clean data set
echo "#### online chain training  ###########"
## check point: sudo nvidia-smi --compute-mode=3 if you have multiple GPU's
#local/chain/run_tdnn_1a.sh
#local/chain/run_tdnn_1b.sh
#local/chain/multi_condition/run_tdnn_lstm_1e.sh --nj $nCPU
local/chain/multi_condition/run_tdnn_1n.sh --nj $nCPU
#local/chain/run_tdnn_opgru_1c.sh --nj $nCPU


finishTime=$(date +'%F-%H-%M')
echo "DNN trainig is finished at" $finishTime
echo "started at" $startTime
echo "finished at" $finishTime
exit 0;

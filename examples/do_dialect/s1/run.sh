#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
echo "Parsed device_ids: ${device_ids[@]}"

stage=8
stop_stage=8

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

# Use your own data path. You need to download the WenetSpeech dataset by yourself.
wenetspeech_data_dir=/ssd/nfs07/binbinzhang/wenetspeech
# Make sure you have 1.2T for ${shards_dir}
shards_dir=/data/duhu/shards/multi_dialect_s1

# control_symbols="<AnHui>,<SiChuan>,<Yue_GZ>,<YunNan>,<GuiZhou>,<NingXia>,<ZangYu>,<WenZhou>,<HuNan>,<HuBei>,<WeiYu>,<HeBei>,<JiangXi>,<Yue_HK>,<LiaoNing>,<GanSu>,<ShanXi>,<ShanDong>,<GuangDong>,<TianJin>,<WuYu>,<MinNan>,<ShangHai>,<FuJian>,<HeNan>"
control_symbols="<SHANDONG>,<WEIYU>,<SHANGHAI>,<FUJIAN>,<ANHUI>,<JIANGXI>,<HEBEI>,<GUIZHOU>,<LIAONING>,<MINNAN>,<HENAN>,<ZANGYU>,<GANSU>,<YUNNAN>,<HUBEI>,<SICHUAN>,<NINGXIA>,<YUE_HK>,<WUYU>,<HUNAN>,<WENZHOU>,<GUANGDONG>,<YUE_GZ>,<SHANXI>,<TIANJIN>,<MANDARIN>"
# WenetSpeech training set

train_set=train
dev_set=dev
test_sets="kespeech King-ASR-345 King-ASR-384-15-Shaanxi King-ASR-384-2 King-ASR-384-4-Jiangxi King-ASR-406 King-ASR-424 King-ASR-668 King-ASR-853-1 King-ASR-882 King-ASR-384-10-Tianjin King-ASR-384-16-Shanxi King-ASR-384-20-Hebei King-ASR-384-6-WU King-ASR-407 King-ASR-426 King-ASR-669 King-ASR-854 King-ASR-883 King-ASR-070 King-ASR-384-11-Anhui King-ASR-384-17-Hubei King-ASR-384-21-Liaoning King-ASR-384-7-Yunnan King-ASR-420 King-ASR-427 King-ASR-742 King-ASR-854-1 King-ASR-083 King-ASR-384-12-Shandong King-ASR-384-18-Gansu King-ASR-384-22-Wu King-ASR-384-8-Guizhou King-ASR-421 King-ASR-443 King-ASR-758 King-ASR-879 King-ASR-086 King-ASR-384-13-Henan King-ASR-384-19-Wenzhou King-ASR-384-23-Ningxia King-ASR-384-9-Wu King-ASR-422 King-ASR-446 King-ASR-791 King-ASR-880 King-ASR-241 King-ASR-384-14-Liaoning King-ASR-384-1-Fujian King-ASR-384-3-Hunan King-ASR-399 King-ASR-423 King-ASR-449 King-ASR-853 King-ASR-881 King-ASR-462 King-ASR-113 King-ASR-722 King-ASR-166"
ke_testset="kespeech"
# test_sets="kespeech"
m_test_sets="ZangYu___MANDARIN"
m_test_sets="ZangYu___MANDARIN LiaoNing___AnHui LiaoNing___GanSu LiaoNing___HuBei SiChuan___Yue_HK"

other_test_sets="test_SPEECHIO_ASR_ZH00001 test_SPEECHIO_ASR_ZH00002 test_SPEECHIO_ASR_ZH00003 test_SPEECHIO_ASR_ZH00004 test_SPEECHIO_ASR_ZH00005 test_SPEECHIO_ASR_ZH00006 test_SPEECHIO_ASR_ZH00007 test_SPEECHIO_ASR_ZH00008 test_SPEECHIO_ASR_ZH00009 test_SPEECHIO_ASR_ZH00010 test_SPEECHIO_ASR_ZH00011 test_SPEECHIO_ASR_ZH00012 test_SPEECHIO_ASR_ZH00013 test_SPEECHIO_ASR_ZH00014 test_SPEECHIO_ASR_ZH00015 test_ws_meeting test_ws_net"

train_config=conf/train_conformer_bidecoder.yaml
dir=exp/v2_unigram_10000_without_lang_tag_stream_pretrained_on_nonstream # clean and more data
checkpoint=exp/v2_unigram_10000_without_lang_tag_pretrained_on_langtag/avg3_modestep_max27999_20240722_03-24-07.pt
tensorboard_dir=tensorboard
num_workers=8
prefetch=10

cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn

# dir=exp/v2_unigram_10000_without_lang_tag_pretrained_on_langtag/
# decode_checkpoint=exp/v2_unigram_10000_without_lang_tag_pretrained_on_langtag/avg3_modestep_max27999_20240722_03-24-07.pt
decode_checkpoint=exp/v2_unigram_10000_without_lang_tag_stream_pretrained_on_nonstream/step_32999.pt
average_checkpoint=false
average_num=3
average_mode=step
max_step=27999
now=`date '+%Y%m%d_%H-%M-%S'`
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_modes="attention attention_rescoring"
train_engine=torch_ddp

deepspeed_config=../whisper/conf/ds_stage1.json
deepspeed_save_states="model+optimizer"

vocab_size=10000
model_type=unigram
dict=data/dict/dict_${vocab_size}.txt
bpemodel=$(dirname $dict)/MDB_${model_type}_${vocab_size}_control # multi-dialect bpe model => mdb

decoding_chunk_size=
ctc_weight=0.5
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
decode_batch=8

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation"
  ./local/data_prep.sh|| exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Generate train and test sets"
  projs=`ls data | grep "King"`
  train_folders=""
  test_folders=""
  for proj in $projs; do 
    train_folders="$train_folders data/$proj/train"
    test_folders="$test_folders data/$proj/test"
  done
  echo $train_folders
  echo $test_folders
  tools/combine_data.sh data/train $train_folders
  tools/combine_data.sh data/dev $test_folders
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Make a dictionary"
    dict_dir=`dirname $dict`
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    echo "<sos/eos> 2" >> $dict
    cut -f 2- -d" " data/${train_set}/text > $dict_dir/input.txt
    ./tools/spm_train --input=$dict_dir/input.txt --vocab_size=${vocab_size} --model_type=${model_type} --model_prefix=${bpemodel} --input_sentence_size=100000000  --control_symbols=$control_symbols
    tools/spm_encode --model=${bpemodel}.model --output_format=piece \
      < $dict_dir/input.txt | \
      tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> ${dict}

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  full_size=`cat data/${train_set}/wav.scp | wc -l`
  sampling_size=$((full_size / cmvn_sampling_divisor))
  shuf -n $sampling_size data/$train_set/wav.scp \
    > data/$train_set/wav.scp.sampled
  python3 tools/compute_cmvn_stats.py \
  --num_workers 16 \
  --train_config $train_config \
  --in_scp data/$train_set/wav.scp.sampled \
  --out_cmvn data/$train_set/global_cmvn
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'
  echo -e "It requires ${RED}1.2T ${NOCOLOR}space for $shards_dir, please make sure you have enough space"
  echo -e "It takes about ${RED}12 ${NOCOLOR}hours with 32 threads"
  for x in $dev_set ${train_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 5000 \
      --num_threads 16 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text \
      $(realpath $dst) data/$x/data.list
  done

  for x in $ke_testset; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 5000 \
      --num_threads 16 --segments data/$x/test/segments \
      data/$x/test/wav.scp data/$x/test/text \
      $(realpath $dst) data/$x/test/data.list
  done

  for x in $test_sets; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 5000 \
      --num_threads 16 --segments data/$x/test/segments \
      data/$x/test/wav.scp data/$x/test/text \
      $(realpath $dst) data/$x/test/data.list
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Start training"
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=2024 --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type "shard" \
      --train_data data/$train_set/data.list \
      --cv_data data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 1200 \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}_${now}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --mode ${average_mode} \
        --max_step ${max_step} \
        --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  i=0
  for testset in ${test_sets}; do # test_sets
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    device_id=${device_ids[i % ${#device_ids[@]}]}
    # export CUDA_VISIBLE_DEVICES="0"
    # device_id=0
    echo "Testing ${testset} on GPU ${device_id}"
    python wenet/bin/recognize.py --gpu ${device_id} \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type "shard" \
      --test_dat data/$testset/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size ${decode_batch} \
      --blank_penalty ${blank_penalty} \
      --length_penalty ${length_penalty} \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_dir $result_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} &
    ((i++))
    if [[ $device_id -eq $((num_gpus - 1)) ]]; then
      wait
    fi
  }
  done
  wait
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        data/$testset/test/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
  }
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model you want"
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}_${now}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --mode ${average_mode} \
        --max_step ${max_step} \
        --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  i=0
  for testset in ${m_test_sets}; do # test_sets
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    # device_id=${device_ids[i % ${#device_ids[@]}]}
    export CUDA_VISIBLE_DEVICES="0"
    device_id=0
    echo "Testing ${testset} on GPU ${device_id}"
    python wenet/bin/recognize.py --gpu ${device_id} \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type "shard" \
      --test_dat data/fake/$testset/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size ${decode_batch} \
      --blank_penalty ${blank_penalty} \
      --length_penalty ${length_penalty} \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_dir $result_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} &
    ((i++))
    wait
    if [[ $device_id -eq $((num_gpus - 1)) ]]; then
      wait
    fi
  }
  done
  wait
  for testset in ${m_test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        data/fake/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
  }
  done
fi




if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}_${now}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --mode ${average_mode} \
        --max_step ${max_step} \
        --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  i=0
  for testset in ${other_test_sets}; do # test_sets
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    # device_id=${device_ids[i % ${#device_ids[@]}]}
    export CUDA_VISIBLE_DEVICES="0"
    device_id=0
    echo "Testing ${testset} on GPU ${device_id}"
    python wenet/bin/recognize.py --gpu ${device_id} \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type "shard" \
      --test_dat data/$testset/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size ${decode_batch} \
      --blank_penalty ${blank_penalty} \
      --length_penalty ${length_penalty} \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_dir $result_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    ((i++))
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        data/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
  }
  done
  wait
fi
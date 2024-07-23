#!/bin/bash

data_dir=data/fake
shards_dir=/data/duhu/shards/multi_dialect_s1

while true; do
    for x in `ls $data_dir`; do
        sub_dir=$data_dir/$x
        if [ ! -f $data_dir/.${x}.done ]; then
            continue
        fi
        if [ -f ${x}.done ]; then
            continue
        fi
        dst=$shards_dir/$x
        echo $sub_dir
        mkdir -p $dst
        tools/make_shard_list.py --resample 16000 --num_utts_per_shard 5000 \
        --num_threads 16 \
        $sub_dir/wav.scp $sub_dir/text \
        $(realpath $dst) $sub_dir/data.list
        touch ${x}.done
    done
    sleep 100
done
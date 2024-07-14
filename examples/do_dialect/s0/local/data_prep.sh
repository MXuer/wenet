#!/bin/bash

conf_file=conf/do_data.yaml
tasks="King-ASR-426 King-ASR-384-14-Liaoning King-ASR-406 King-ASR-399 King-ASR-384-1-Fujian King-ASR-345 King-ASR-853-1 King-ASR-384-13-Henan King-ASR-384-19-Wenzhou King-ASR-086 King-ASR-407 King-ASR-384-3-Hunan King-ASR-742 King-ASR-384-23-Ningxia King-ASR-384-21-Liaoning King-ASR-384-15-Shaanxi King-ASR-083 King-ASR-384-11-Anhui King-ASR-443 King-ASR-881 King-ASR-423 King-ASR-422 King-ASR-241 King-ASR-384-10-Tianjin King-ASR-446 King-ASR-384-2 King-ASR-882 King-ASR-854 King-ASR-384-22-Wu King-ASR-424 King-ASR-070 King-ASR-854-1 King-ASR-879 King-ASR-384-12-Shandong King-ASR-420 King-ASR-758 King-ASR-384-18-Gansu King-ASR-427 King-ASR-384-20-Hebei King-ASR-384-6-WU King-ASR-384-4-Jiangxi King-ASR-791 King-ASR-384-7-Yunnan King-ASR-384-17-Hubei King-ASR-384-16-Shanxi King-ASR-384-8-Guizhou King-ASR-880 King-ASR-883 King-ASR-384-9-Wu King-ASR-669 King-ASR-668 King-ASR-449 King-ASR-421 King-ASR-853"
for task in $tasks; do
    echo "TASK : $task...."
    python local/data_prep.py --task $task --conf-file $conf_file || exit 1;
    if [ -f data/.${task}.done ]; then
        continue
    fi
    ./tools/fix_data_dir.sh data/$task/train
    ./tools/fix_data_dir.sh data/$task/test
done
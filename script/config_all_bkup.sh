#!/bin/bash

# This script is used to generate all dataset configs.

module load python/3 gcc/7.2.0
export PYTHONPATH=/home/jinhuis2/scratch/mypython3:$PYTHONPATH

CONFIG="python3 config_generator.py"
N_RUN=12
MODE=$1

if [[ $MODE == "train" ]]; then
    $CONFIG -t cb-tleft-small-bkup -lb 2 2 -rb 2 3 -m -n $N_RUN -e 50 -p left-btnk-bkup
    $CONFIG -t cb-tleft-middle-bkup -lb 2 2 3 4 -rb 5 6 4 4 -m -n $N_RUN -e 50 -p left-btnk-bkup
    $CONFIG -t cb-tright-small-bkup -lb 2 2 -rb 2 3 -m -n $N_RUN -e 50 -p right-btnk-bkup
    $CONFIG -t cb-tright-middle-bkup -lb 2 2 3 4 -rb 5 6 4 4 -m -n $N_RUN -e 50 -p right-btnk-bkup
    # $CONFIG -t cbtnk-large -lb 2 4 -rb 8 16 -n $N_RUN -e 50
elif [[ $MODE == "test" ]]; then
    $CONFIG -t cbtest-one-to-n-bkup -p one-to-n-bkup -e 50        # 12 runs, >12h
#    $CONFIG -t cbtest-path-lag-bkup -p path-lag-bkup -e 50        # 24 runs, 4h
    $CONFIG -t cbtest-load-scan-bkup -p load-scan-bkup -e 50      # 20 runs, >12h
    $CONFIG -t cbtest-large-flow-bkup -p large-flow-bkup -e 50    # 24 runs, 3d
    $CONFIG -t cbtest-para-btnk-bkup -p para-btnk-bkup -e 50      # 16 runs, 3d
else
    echo "Invalid mode: $MODE"
fi

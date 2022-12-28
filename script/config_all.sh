#!/usr/bash

# This script is used to generate all dataset configs.

module load python/3 gcc/7.2.0
export PYTHONPATH=/home/jinhuis2/scratch/mypython3:$PYTHONPATH

CONFIG="python3 config_generator.py"
N_RUN=36
MODE=TRAIN

if [[ $MODE == "TRAIN"]]; then
    $CONFIG -t cb-tleft-small -lb 2 2 -rb 2 3 -m -n $N_RUN -e 50 -p left-btnk
    $CONFIG -t cb-tleft-middle -lb 2 2 3 4 -rb 5 6 4 4 -m -n $N_RUN -e 50 -p left-btnk
    $CONFIG -t cb-tright-small -lb 2 2 -rb 2 3 -m -n $N_RUN -e 50 -p right-btnk
    $CONFIG -t cb-tright-middle -lb 2 2 3 4 -rb 5 6 4 4 -m -n $N_RUN -e 50 -p right-btnk
    $CONFIG -t cbtnk-large -lb 2 4 -rb 8 16 -n $N_RUN -e 50
elif [[ $MODE == "TEST" ]]; then
    $CONFIG -t cbtest-one-to-n -p one-to-n -e 50        # 12 runs, >12h
    $CONFIG -t cbtest-path-lag -p path-lag -e 50        # 24 runs, 4h
    $CONFIG -t cbtest-load-scan -p load-scan -e 50      # 20 runs, >12h
    $CONFIG -t cbtest-large-flow -p large-flow -e 50    # 24 runs, 3d
    $CONFIG -t cbtest-para-btnk -p para-btnk -e 50      # 16 runs, 3d
else
    echo "Invalid mode: $MODE"
fi

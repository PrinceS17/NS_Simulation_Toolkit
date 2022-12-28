#!/bin/bash

CSL_SETTING="--time=3-00:00:00 --partition=csl"
ENG_SETTING="--time=12:00:00 --partition=eng-instruction"
SEC_SETTING="--time=4:00:00 --partition=secondary"

TRAIN_DIR="train_tailored"
TEST_DIR="test_basic"
#POST_CUT='cut -f 4 -d " "'

if [[ $1 == "train" ]]; then
    # 4 x 36 runs
    JOB_TL_S1=$(sbatch ${SEC_SETTING} ${TRAIN_DIR}/left_small.sbatch | cut -f 4 -d " ")
    JOB_TL_S2=$(sbatch --dependency=afterany:${JOB_TL_S1} ${SEC_SETTING} ${TRAIN_DIR}/left_small.sbatch | cut -f 4 -d " ")
    JOB_TR_S1=$(sbatch --dependency=afterany:${JOB_TL_S2} ${SEC_SETTING} ${TRAIN_DIR}/right_small.sbatch | cut -f 4 -d " ")
    JOB_TR_S2=$(sbatch --dependency=afterany:${JOB_TR_S1} ${SEC_SETTING} ${TRAIN_DIR}/right_small.sbatch | cut -f 4 -d " ")

elif [[ $1 == "test" ]]; then
    # test runs
    sbatch ${CSL_SETTING} ${TEST_DIR}/large_flow_L.sbatch
    sbatch ${CSL_SETTING} ${TEST_DIR}/para_btnk_L.sbatch
    sbatch ${CSL_SETTING} ${TEST_DIR}/load_scan_L.sbatch
    sbatch ${CSL_SETTING} ${TEST_DIR}/one_to_n_L.sbatch
    sbatch ${ENG_SETTING} ${TEST_DIR}/path_lag_M.sbatch
fi

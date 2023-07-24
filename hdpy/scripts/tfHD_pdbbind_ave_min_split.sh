#!/usr/bin/bash


N_SAMPLES=1
TRAIN_DATA=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/ave_min_split/ave_min_split_train.npy
TEST_DATA=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/ave_min_split/ave_min_split_test.npy
OUTPUT_PREFIX=train_pdbbind_ave_min_split
OUTPUT_HYPEROPT_DATA_CSV=../../results/pdbbind/${OUTPUT_PREFIX}_hyperopt_results.csv
RANDOM_SEED=1
TF_DEVICE_STR=/GPU:0
# python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --random-seed $RANDOM_SEED

python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --output-prefix $OUTPUT_PREFIX
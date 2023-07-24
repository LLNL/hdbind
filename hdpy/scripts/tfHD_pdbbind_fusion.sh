#!/usr/bin/bash


N_SAMPLES=1
TRAIN_DATA=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/train_fusion.npy
TEST_DATA=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/test_fusion.npy
OUTPUT_PREFIX=train_pdbbind_refined_2019_test_pdbbind_casf_2016_fusion
OUTPUT_HYPEROPT_DATA_CSV=../../results/pdbbind/${OUTPUT_PREFIX}_hyperopt_results.csv
RANDOM_SEED=1
TF_DEVICE_STR=/GPU:0
# python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --random-seed $RANDOM_SEED

python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --output-prefix $OUTPUT_PREFIX
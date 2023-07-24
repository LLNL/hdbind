#!/usr/bin/bash


N_SAMPLES=1
TRAIN_DATA=../../datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/refined.npy
TEST_DATA=../../datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/casf-2016.npy
OUTPUT_PREFIX=train_pdbbind_refined_2019_test_pdbbind_casf_2016_morgan_fp
OUTPUT_HYPEROPT_DATA_CSV=../../results/pdbbind/${OUTPUT_PREFIX}_hyperopt_results.csv
RANDOM_SEED=1
TF_DEVICE_STR=/GPU:0
# python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --random-seed $RANDOM_SEED

python tfHD_pdbbind.py --train-data $TRAIN_DATA --test-data $TEST_DATA --n-samples $N_SAMPLES -O $OUTPUT_HYPEROPT_DATA_CSV --tf-device-str $TF_DEVICE_STR --output-prefix $OUTPUT_PREFIX
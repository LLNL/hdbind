#!/bin/bash

DATASET="sider"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=4

# ECFP encoding and Random Projection of ECFP feature
for model in "ecfp" "rp";
do
    python hd_main.py --dataset $DATASET --split-type $fold --model $model --tokenizer atomwise --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
done

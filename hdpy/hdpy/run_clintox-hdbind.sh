#!/bin/bash

DATASET="clintox"
D=10000
N_TRIALS=1
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=123

# ECFP encoding and Random Projection of ECFP feature
for model in "ecfp" "rp" "selfies"; 
do
    for split in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $split --model $model --tokenizer atomwise --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done
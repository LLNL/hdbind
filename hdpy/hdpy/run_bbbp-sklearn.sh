#!/bin/bash

DATASET="bbbp"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=125

# sklearn models
for model in "rf" "mlp"; 
do
    for split in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $split --model $model --random-state $RANDOM_STATE --n-trials $N_TRIALS
    done
done
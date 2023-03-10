#!/bin/bash

# DUD-E only has a random split defined 

DATASET="lit-pcba"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=4


# sklearn models
for model in "rf" "mlp";
do
    python hd_main.py --dataset $DATASET --split-type random --model $model --n-trials $N_TRIALS --random-state $RANDOM_STATE

done


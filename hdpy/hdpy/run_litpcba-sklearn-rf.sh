#!/bin/bash

# DUD-E only has a random split defined 

DATASET="lit-pcba"
D=1000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=125


# sklearn models
for model in "rf"; 
do
    python hd_main.py --cpu-only --dataset $DATASET --split-type random --model $model --n-trials $N_TRIALS --random-state $RANDOM_STATE
done


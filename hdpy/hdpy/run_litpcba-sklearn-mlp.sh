#!/bin/bash

# DUD-E only has a random split defined 

DATASET="lit-pcba"
D=1000
N_TRIALS=1
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=123


# sklearn models
for model in "mlp"; 
do
    python hd_main.py --dataset $DATASET --split-type random --model $model --n-trials $N_TRIALS --random-state $RANDOM_STATE
done


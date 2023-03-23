#!/bin/bash

DATASET="clintox"
D=10000
N_TRIALS=1
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=123

# sklearn models
for model in "rf" "mlp"; 
do
    for split in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $split --model $model --random-state $RANDOM_STATE --n-trials $N_TRIALS
    done
done

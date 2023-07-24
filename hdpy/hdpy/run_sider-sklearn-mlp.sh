#!/bin/bash

DATASET="sider"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=125

# sklearn models
for model in "mlp"; 
do
    for fold in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $fold --model $model --random-state $RANDOM_STATE --n-trials $N_TRIALS
    done
done

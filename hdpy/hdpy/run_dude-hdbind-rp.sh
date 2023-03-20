#!/bin/bash

# DUD-E only has a random split defined 

DATASET="dude"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=4

# ECFP encoding and Random Projection of ECFP feature
for model in "rp"; 
do
    python hd_main.py --dataset $DATASET --split-type random --model $model --tokenizer atomwise --D $D --n-trials $N_TRIALS --hd-retrain-epochs $HD_RETRAIN_EPOCHS --random-state $RANDOM_STATE
done
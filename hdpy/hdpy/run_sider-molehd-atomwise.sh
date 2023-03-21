#!/bin/bash

DATASET="sider"
D=10000
N_TRIALS=3
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=3

# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "atomwise"; 
do
    for fold in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $fold --model smiles-pe --tokenizer $tokenizer --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done
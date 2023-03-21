#!/bin/bash

DATASET="lit-pcba"
D=10000
N_TRIALS=3
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=3

# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "atomwise"; 
do
    python hd_main.py --dataset $DATASET --split-type random --model smiles-pe --tokenizer $tokenizer --D $D --n-trials $N_TRIALS --hd-retrain-epochs $HD_RETRAIN_EPOCHS --random-state $RANDOM_STATE
done
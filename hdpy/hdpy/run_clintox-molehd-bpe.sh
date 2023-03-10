#!/bin/bash

DATASET="clintox"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=4


# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "bpe" "atomwise";
do
    for fold in "random" "scaffold";
    do
        python hd_main.py --dataset $DATASET --split-type $fold --model smiles-pe --tokenizer $tokenizer --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done

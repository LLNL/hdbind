#!/bin/bash

DATASET="clintox"
D=10000
N_TRIALS=1
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=123


# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "bpe" "atomwise"; 
do
    for split in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $split --model smiles-pe --tokenizer $tokenizer --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done

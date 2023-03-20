#!/bin/bash

DATASET="sider"
D=10000
N_TRIALS=10
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=4

# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "1"; 
do
    for fold in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $fold --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done

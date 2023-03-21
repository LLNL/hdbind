#!/bin/bash

# DUD-E only has a random split defined 

DATASET="dude"
D=10000
N_TRIALS=3
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=3

# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "3"; 
do
    python hd_main.py --dataset $DATASET --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --D $D --n-trials $N_TRIALS --hd-retrain-epochs $HD_RETRAIN_EPOCHS --random-state $RANDOM_STATE
done

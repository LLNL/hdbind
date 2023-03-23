#!/bin/bash

DATASET="lit-pcba"
D=1000
N_TRIALS=1
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=123

# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "3"; 
do
    python hd_main.py --dataset $DATASET --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --D $D --n-trials $N_TRIALS --hd-retrain-epochs $HD_RETRAIN_EPOCHS --random-state $RANDOM_STATE
done
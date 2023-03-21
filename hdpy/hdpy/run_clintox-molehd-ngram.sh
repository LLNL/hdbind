#!/bin/bash

DATASET="clintox"
D=10000
N_TRIALS=3
HD_RETRAIN_EPOCHS=10
RANDOM_STATE=3

# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "1" "2" "3"; 
do
    for split in "random" "scaffold"; 
    do
        python hd_main.py --dataset $DATASET --split-type $split --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state $RANDOM_STATE --hd-retrain-epochs $HD_RETRAIN_EPOCHS --n-trials $N_TRIALS
    done
done

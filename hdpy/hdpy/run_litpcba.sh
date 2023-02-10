#!/bin/bash

# we just use a random split for lit-pcba since we have an already defined AVE split we can use 

# ECFP encoding and Random Projection of ECFP feature
for model in "ecfp" "rp";
do

    python hd_main.py --dataset lit-pcba --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

done

# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "bpe" "atomwise";
do

    python hd_main.py --dataset lit-pcba --split-type random --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10

done


# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "1" "2" "3";
do

    python hd_main.py --dataset lit-pcba --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10

done


# sklearn models
for model in "rf" "mlp";
do

    python hd_main.py --dataset lit-pcba --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

done


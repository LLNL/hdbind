#!/bin/bash

# ECFP encoding and Random Projection of ECFP feature
for model in "ecfp" "rp";
do
    python hd_main.py --dataset clintox --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset clintox --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset bbbp --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset bbbp --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset sider --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset sider --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
done

# smiles-pe models with byte-pair-encoding, smiles-pe atomwise tokenizer
for tokenizer in "bpe" "atomwise";
do
    python hd_main.py --dataset clintox --split-type random --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset clintox --split-type scaffold --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset bbbp --split-type random --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset bbbp --split-type scaffold --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset sider --split-type random --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset sider --split-type scaffold --model smiles-pe --tokenizer $tokenizer --random-state 0 --hd-retrain-epochs 10
done


# smiles-pe ngram tokenizer, unigram + bigram + trigram
for ngram_order in "1" "2" "3";
do
    python hd_main.py --dataset clintox --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset clintox --split-type scaffold --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset bbbp --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset bbbp --split-type scaffold --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset sider --split-type random --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset sider --split-type scaffold --model smiles-pe --tokenizer ngram --ngram-order $ngram_order --random-state 0 --hd-retrain-epochs 10
done


# sklearn models
for model in "rf" "mlp";
do
    python hd_main.py --dataset clintox --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset clintox --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset bbbp --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset bbbp --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10

    python hd_main.py --dataset sider --split-type random --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
    python hd_main.py --dataset sider --split-type scaffold --model $model --tokenizer atomwise --random-state 0 --hd-retrain-epochs 10
done

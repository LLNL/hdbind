#!/usr/bin/bash


NUM_EPOCHS=100
#for data_path in datasets/dude/deepchem_feats/*/rdkit/;
#for data_path in /g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/data_utils/debug/*/mol2vec/;
for data_path in /usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/data_utils/mol2vec/*/mol2vec;
do
	echo "python experiment_full_dataset_dude.py --out-csv $data_path/results.csv --dataset dude --model-list HD --train-path-list $data_path/train.npy --test-path-list $data_path/test.npy --num-epochs $NUM_EPOCHS"

	# python experiment_full_dataset_dude.py --dataset dude --model-list HD --train-path-list datasets/dude/deepchem_feats/p*/rdkit/train.npy --test-path-list datasets/dude/deepchem_feats/p*/rdkit/test.npy --num-epochs 100 --out-csv debug_full_dude.csv --hidden-size 1000 --lr 1 --n-problems 1
done

#!/usr/bin/bash

python encode_smiles.py --ngram-order 2 --dataset bbbp
python encode_smiles.py --ngram-order 2 --dataset clintox
#python convert_smiles_to_bpe.py --ngram-order 2 --dataset sider 

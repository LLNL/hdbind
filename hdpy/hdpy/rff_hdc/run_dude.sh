#!/usr/bin/bash

for path in dataset/dude-mol2vec/*/mol2vec;
do
	echo $path
	python main.py -gamma 0.3 -epoch 100 -gorder 8 -dim 10000 -dataset dude --dude-path $path -model rff-hdc
done
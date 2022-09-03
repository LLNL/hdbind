#!/usr/bin/bash
export AMPLXE_RUNTOOL_OPTIONS=--no-altstack

N_PROBLEMS=30
K_QUERY=-1

SUPPORT_PATH=datasets/postera/rdkit/train.npy
QUERY_PATH=datasets/postera/rdkit/test.npy
OUTPUT_DIR=results/postera

for feat_type in "ecfp" "smiles_to_seq" "smiles_to_image" "mordred" "maacs" "rdkit";
do
	output_path="${OUTPUT_DIR}/${feat_type}"
	python experiment_single_classifier.py --dataset postera --out_csv "${output_path}/test_results.csv" --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY
done
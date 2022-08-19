#!/usr/bin/bash

INPUT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/raw_data/merged_mpro_exp_data2_train_valid_test_ave_min_8aa6f8e2-4acc-461e-b8aa-9917eac1a337.csv
OUTPUT_DIR=debug
SMILES_COL=rdkit_smiles 
LABEL_COL=active


#for feat_type in "ecfp" "smiles_to_seq" "smiles_to_image" "mordred" "maacs" "rdkit";
# for feat_type in "coul_matrix";
for feat_type in "mol2vec";
do
	echo $feat_type
	python feat.py --input-path $INPUT_PATH --smiles-col $SMILES_COL --label-col $LABEL_COL --feat-type $feat_type --output-dir "${OUTPUT_DIR}/${feat_type}"
done

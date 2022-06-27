#!/usr/bin/bash


# python feat.py --input-path /usr/workspace/atom/PostEra/eval/mpro_exp_data2.csv --smiles-col rdkit_smiles --label-col active

INPUT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/metadata/pdbbind_2019_metadata_full_with_smiles_sanitized.csv
OUTPUT_DIR=datasets/pdbbind_2019
SMILES_COL=smiles 
LABEL_COL="affinity"

feat_type="ecfp"
# for feat_type in "ecfp" "smiles_to_seq" "smiles_to_image" "mordred" "maacs" "rdkit";
# for feat_type in "coul_matrix";
# do
	# echo $feat_type
python feat.py --input-path $INPUT_PATH --smiles-col $SMILES_COL --label-col $LABEL_COL --feat-type $feat_type --output-dir "${OUTPUT_DIR}/${feat_type}"
# done

#!/usr/bin/bash

# INPUT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/raw_data/merged_mpro_exp_data2_train_valid_test_ave_min_8aa6f8e2-4acc-461e-b8aa-9917eac1a337.csv
OUTPUT_DIR=mol2vec
SMILES_COL=smiles 
LABEL_COL=decoy
NUM_WORKERS=35

#for feat_type in "ecfp" "smiles_to_seq" "smiles_to_image" "mordred" "maacs" "rdkit";
# for feat_type in "coul_matrix";
# for input_path in /usr/workspace/atom/gbsa_modeling/dude_smiles/*_gbsa_smiles.csv;
# do
    # target=$(basename $input_path | cut -d '_' -f1)
for feat_type in "mol2vec";
do
    echo $feat_type
    python feat.py --num-workers $NUM_WORKERS --input-path-list /usr/workspace/atom/gbsa_modeling/dude_smiles/*_gbsa_smiles.csv --smiles-col $SMILES_COL --label-col $LABEL_COL --feat-type $feat_type --output-dir "${OUTPUT_DIR}" --invert-labels
done
# done

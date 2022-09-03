#!/usr/bin/bash
export AMPLXE_RUNTOOL_OPTIONS=--no-altstack

N_PROBLEMS=30
K_QUERY=-1

cd /g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/fsl

for target in $(find datasets/dude/deepchem_feats -type d \( ! -name deepchem_feats \) -maxdepth 1 -mindepth 1 -printf "%f\n");
do
    for feat_type in "ecfp" "smiles_to_seq" "smiles_to_image" "mordred" "maacs" "rdkit";
    do
        SUPPORT_PATH=datasets/dude/deepchem_feats/$target/$feat_type/train.npy
        QUERY_PATH=datasets/dude/deepchem_feats/$target/$feat_type/test.npy 
        OUTPUT_DIR=results/dude/

       output_path=${OUTPUT_DIR}/${target}/${feat_type}/test_results.csv
       echo "python experiment_single_classifier.py --dataset dude --out_csv ${output_path} --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY"
    done
done 

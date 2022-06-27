#!/usr/bin/bash
export AMPLXE_RUNTOOL_OPTIONS=--no-altstack

N_PROBLEMS=30
K_QUERY=-1

# fusion features
# SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/train_fusion.npy
# SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/train.npy
# QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/test.npy
# QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/train.npy
# SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/refined.npy
# QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/casf-2016.npy
# QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/test_fusion.npy
# SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/data_utils/datasets/postera/smiles_to_seq/train.npy
# QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/data_utils/datasets/postera/smiles_to_seq/test.npy
# SUPPORT_PATH=datasets/pdbbind/fusion_ml/fusion/train_fusion.npy
# QUERY_PATH=datasets/pdbbind/fusion_ml/fusion/test_fusion.npy
#SUPPORT_PATH=datasets/postera/smiles_to_image/train.npy
#QUERY_PATH=datasets/postera/smiles_to_image/test.npy

SUPPORT_PATH=datasets/postera/rdkit/train.npy
QUERY_PATH=datasets/postera/rdkit/test.npy


# vtune --collect hpc-performance python experiment_single_classifier.py --dataset pdbbind --out_csv debug_results.csv --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY
python experiment_single_classifier.py --dataset postera --out_csv debug_results.csv --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY
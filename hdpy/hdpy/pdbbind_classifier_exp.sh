#!/usr/bin/bash

N_PROBLEMS=50
K_QUERY=20

# pdbbind fusion features
SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/train_fusion.npy
QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/fusion_ml/fusion/test_fusion.npy
python experiment_single_classifier.py --dataset pdbbind --out_csv fsl_fusion_pdbbind_results.csv --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY

# pdbbind morgan fingerprints
#SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/refined.npy
#QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/pdbbind_2019_fingerprints_with_casf/casf-2016.npy
#python experiment_single_classifier.py --dataset pdbbind --out_csv fsl_morganfp_pdbbind_results.csv --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY

# postera morgan fingerprints
#SUPPORT_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/train.npy
#QUERY_PATH=/g/g13/jones289/workspace/hd-cuda-master/datasets/postera/test.npy
#python experiment_single_classifier.py --dataset postera --out_csv fsl_morganfp_postera_results.csv --n_problems $N_PROBLEMS --support_path $SUPPORT_PATH --query_path $QUERY_PATH --kquery $K_QUERY
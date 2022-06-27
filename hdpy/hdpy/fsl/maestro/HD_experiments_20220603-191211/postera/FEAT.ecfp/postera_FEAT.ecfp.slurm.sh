#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=pvis
#SBATCH --account=ncov2019
#SBATCH --time=12:00:00
#SBATCH --job-name="postera_FEAT.ecfp"
#SBATCH --output="postera_FEAT.ecfp.out"
#SBATCH --error="postera_FEAT.ecfp.err"
#SBATCH --comment "run eval job on configuration"


cd /usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/fsl

DATA_DIR=datasets/postera/ecfp

#python -m torch.utils.bottleneck experiment_single_classifier.py --dataset postera --out_csv results/postera/ecfp_test_results.csv --n_problems 30 --support_path ${DATA_DIR}/train.npy --query_path ${DATA_DIR}/test.npy 
python experiment_single_classifier.py --dataset postera --out_csv results/postera/ecfp_test_results.csv --n_problems 30 --support-path-list ../../../datasets/dude/deepchem_feats/*/ecfp/train.npy --query-path-list ../../../datasets/dude/deepchem_feats/*/ecfp/test.npy --model-list HD HD-Sparse 



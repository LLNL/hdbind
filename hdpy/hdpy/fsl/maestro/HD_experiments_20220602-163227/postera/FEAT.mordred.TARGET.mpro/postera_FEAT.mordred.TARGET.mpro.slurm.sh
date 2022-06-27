#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=pvis
#SBATCH --account=ncov2019
#SBATCH --time=12:00:00
#SBATCH --job-name="postera_FEAT.mordred.TARGET.mpro"
#SBATCH --output="postera_FEAT.mordred.TARGET.mpro.out"
#SBATCH --error="postera_FEAT.mordred.TARGET.mpro.err"
#SBATCH --comment "run eval job on configuration"


cd /usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/fsl

DATA_DIR=datasets/dude/deepchem_feats/mpro/mordred
python experiment_single_classifier.py --dataset postera --out_csv results/postera/mpro_mordred_test_results.csv --n_problems 30 --support_path ${DATA_DIR}/train.npy --query_path ${DATA_DIR}/test.npy --kquery -1


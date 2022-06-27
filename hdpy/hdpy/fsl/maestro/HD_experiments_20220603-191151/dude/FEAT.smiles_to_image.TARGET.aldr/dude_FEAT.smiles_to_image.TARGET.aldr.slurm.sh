#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --account=ncov2019
#SBATCH --time=1-00:00:00
#SBATCH --job-name="dude_FEAT.smiles_to_image.TARGET.aldr"
#SBATCH --output="dude_FEAT.smiles_to_image.TARGET.aldr.out"
#SBATCH --error="dude_FEAT.smiles_to_image.TARGET.aldr.err"
#SBATCH --comment "run eval job on configuration"


cd /usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/fsl

DATA_DIR=datasets/dude/deepchem_feats/aldr/smiles_to_image
python experiment_single_classifier.py --dataset dude --out_csv results/dude/aldr_smiles_to_image_test_results.csv --n_problems 30 --support_path ${DATA_DIR}/train.npy --query_path ${DATA_DIR}/test.npy 


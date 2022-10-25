#!/usr/bin/bash


root_path=datasets/lit_pcba/AVE_unbiased
#root_path=datasets/lit_pcba/lit_pcba_full_data


for path in ${root_path}/*
do
    target=$(basename ${path})
    python feat.py --input-path-list $(find ${root_path}/${target} -name *inactives.smi) --smiles-col 0 --feat-type ecfp --no-subset --num-workers 70 --output-dir ${root_path}/${target}/inactives

    python feat.py --input-path-list $(find ${root_path}/${target} -name *actives.smi) --smiles-col 0 --feat-type ecfp --no-subset --num-workers 70 --output-dir ${root_path}/${target}/actives
done

#!/bin/bash

PART=pbatch
TIME=1-00:00:00

PART=pvis
TIME=12:00:00

printf "submitting hdbind jobs\n"
for file in *dude-hdbind*.sh;
do 
    cmd="sbatch -p $PART -t $TIME $file;"
    printf "$cmd\n"
    eval $cmd

done


printf "submitting mole-hd jobs\n"
for file in run_dude-molehd-atomwise.sh run_dude-molehd-bpe.sh;
do 
    cmd="sbatch -p $PART -t $TIME $file;"
    printf "$cmd\n"
    eval $cmd
done


printf 'submitting sklearn jobs\n'
for file in run_dude-sklearn-mlp.sh run_dude-sklearn-rf.sh;
do 
    cmd="sbatch -p $PART -t $TIME $file;"
    printf "$cmd\n"
    eval $cmd
done

#!/bin/bash


# for file in run_litpcba*.sh;
# do 
    # sbatch -p pbatch -t 1-00:00:00 $file;
    # echo $file 
# done

PART=pbatch
TIME=1-00:00:00

# PART=pvis
# TIME=12:00:00

printf "submitting hdbind jobs\n"
for file in run_litpcba-hdbind-ecfp.sh;
do 
    cmd="sbatch -p $PART -t $TIME $file;"
    printf "$cmd\n"
    eval $cmd

done


# printf "submitting mole-hd jobs\n"
# for file in run_litpcba-molehd-atomwise.sh run_litpcba-molehd-bpe.sh;
# do 
    # cmd="sbatch -p $PART -t $TIME $file;"
    # printf "$cmd\n"
    # eval $cmd
# done


printf 'submitting sklearn jobs\n'
for file in run_litpcba-sklearn-mlp.sh;
do 
    cmd="sbatch -p $PART -t $TIME $file;"
    printf "$cmd\n"
    eval $cmd
done

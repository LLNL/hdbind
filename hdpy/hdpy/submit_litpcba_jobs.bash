#!/bin/bash


for file in run_litpcba*.sh;
do 
    sbatch -p pbatch -t 1-00:00:00 $file;
    # echo $file 
done

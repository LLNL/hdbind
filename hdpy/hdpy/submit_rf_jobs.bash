#!/bin/bash


for file in *rf*.sh;
do 
    # sbatch -p pbatch -t 1-00:00:00 $file;
    echo $file 
done

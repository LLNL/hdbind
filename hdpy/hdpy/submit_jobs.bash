#!/bin/bash


for file in *.sh;
do 
    sbatch -p pbatch -t 1-00:00:00 $file; 
done

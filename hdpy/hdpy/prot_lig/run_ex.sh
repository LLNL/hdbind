for seed in {0..9}
do 
    # python pdbbind_main.py --p 1.0 --model aa_seq_ecfp --D 10000 --seed $seed
    python pdbbind_main.py --p 1.0 --model complex-graph --D 10000 --seed $seed
done

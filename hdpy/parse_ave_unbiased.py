################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
import multiprocessing as mp


def main():

    root_p = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")


    for p in root_p.glob("*"):

        train_files = list(p.glob("*T.smi"))
        test_files = list(p.glob("*V.smi"))

        print(len(train_files), train_files)
        print(len(test_files), test_files)


        train_df_list = []
        for train_p in train_files:

            df = pd.read_csv(train_p, header=None, delim_whitespace=True)

            if "_active" in train_p.name:

                df["label"] = [1] * len(df)

            elif "_inactive" in train_p.name:
                                
                df["label"] = [0] * len(df)

            train_df_list.append(df)

        train_df = pd.concat(train_df_list)


        test_df_list = []
        for test_p in test_files:

            df = pd.read_csv(test_p, header=None, delim_whitespace=True)

            if "_active" in test_p.name:

                df["label"] = [1] * len(df)

            elif "_inactive" in test_p.name:
                                
                df["label"] = [0] * len(df)

            test_df_list.append(df)
        
        test_df = pd.concat(test_df_list)



        full_df = pd.concat([train_df, test_df])


        print(train_df.shape, test_df.shape, full_df.shape)

        out_train_p = p / Path("train_data.csv")
        out_test_p = p / Path("test_data.csv")
        out_full_p = p / Path("full_data.csv")

        print(out_train_p, out_test_p, out_full_p)
        

        train_df.to_csv(out_train_p, index=False)
        test_df.to_csv(out_test_p, index=False)
        full_df.to_csv(out_full_p, index=False)

if __name__ == "__main__":
    main()

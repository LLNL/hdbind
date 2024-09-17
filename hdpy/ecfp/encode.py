################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from tkinter import E
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import time

from hdpy.ecfp import compute_fingerprint_from_smiles


def binarize(hv):
    hv = torch.where(hv > 0, hv, -1)
    hv = torch.where(hv <= 0, hv, 1)

    return hv

class StreamingECFPEncoder(Dataset):
    def __init__(self, D: int, radius: int, input_size: int, smiles_list: list):
        super()
        self.D = D
        self.input_size = input_size
        self.radius = radius
        self.smiles_list = smiles_list
        self.item_mem = None
        self.name = "ecfp"

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.item_mem is None:
            self.build_item_memory()

        ecfp = compute_fingerprint_from_smiles(self.smiles_list[idx], length=self.input_size, radius=self.radius)
        if ecfp is not None:
            # start = time.perf_counter()
            hv = self.encode(ecfp)
            # end = time.perf_counter()

            return hv
        else:
            return None

    def build_item_memory(self):
        self.item_mem = torch.bernoulli(
            torch.empty(self.input_size, 2, self.D).uniform_(0, 1)
        )
        self.item_mem = torch.where(self.item_mem <= 0, self.item_mem, -1).int()

    def encode(self, datapoint, return_time=False):

        if self.item_mem is None:
            print("Build item memory before encoding")

        start = time.perf_counter()
        hv = torch.zeros(self.D, dtype=torch.int)


        for pos, value in enumerate(datapoint):
            hv += self.item_mem[pos, value.item()]

        hv = binarize(hv)
        end = time.perf_counter()

        if return_time:
            return hv, torch.ones(1) * (end - start)
        else:
            return hv

class DirectECFPEncoder(Dataset):
    def __init__(self, D: int, radius: int, input_size: int, smiles_list: list, id_list=None):
        super()
        self.D = D
        self.input_size = input_size
        self.radius = radius
        self.smiles_list = smiles_list
        self.item_mem = None
        self.name = "ecfp"
        self.id_list = id_list



    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.item_mem is None:
            self.build_item_memory()

        smiles = self.smiles_list[idx] 
        ecfp = compute_fingerprint_from_smiles(smiles, length=self.input_size, radius=self.radius)

        hv = None
        if ecfp is not None:

            ecfp = torch.from_numpy(ecfp).int()
            # convert binary 0,1 to -1, 1
            hv = binarize(ecfp).int()
            # return hv

        else:
            # return None
            print(f"ECFP calculation failed for {smiles}, returning zero vector")
            hv = torch.zeros(self.D, dtype=torch.int)
            # return hv

        if self.id_list is not None:
            return hv, self.id_list[idx]
        else:
            return hv

    def build_item_memory(self):
        pass

    def encode(self, datapoint):

        pass






def time_ecfp_encoder():
    import pandas as pd
    smiles_df = pd.read_csv("/p/vast1/jones289/lit_pcba/AVE_unbiased/VDR/smiles_test.csv", header=None)

    enc = StreamingECFPEncoder(D=10000, radius=1, input_size=1024, smiles_list=smiles_df[0].values.tolist())
    enc.build_item_memory()

    ecfp_list = [compute_fingerprint_from_smiles(x, length=1024, radius=1).reshape(1,-1) for x in tqdm(smiles_df[0].values.tolist())]
    # import pdb
    # pdb.set_trace()
    ecfp_array = np.concatenate(ecfp_list)
    ecfp_array = torch.from_numpy(ecfp_array)


    data = [enc.encode(x) for x in tqdm(ecfp_array)]

    times = [x[1] for x in data]

    print(f"encoding took {np.mean(times)} seconds.")


def time_direct_ecfp_encoder():
    import pandas as pd
    smiles_df = pd.read_csv("/p/vast1/jones289/lit_pcba/AVE_unbiased/VDR/smiles_test.csv", header=None).sample(frac=0.1)


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecfp-length", type=int)
    parser.add_argument("--ecfp-radius", type=int)
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()


    if args.run_all:
        data_dict ={"length": [], "radius": [], "time_sum": [], "time_mean": [], "sparsity": [], "non_zero":[]}

        for length in [100, 1000, 10000, 100000, 1000000]:
            for radius in [1,2,4,8]:

                ecfp_list = [compute_fingerprint_from_smiles(x, length=length, radius=radius, return_time=True) for x in tqdm(smiles_df[0].values.tolist())]

                ecfp_array = np.vstack([x[0] for x in ecfp_list])

                sum_fp_time = np.sum([x[1] for x in ecfp_list])
                mean_fp_time = np.mean([x[1] for x in ecfp_list])
                sparsity = np.mean(np.sum(ecfp_array, axis=1) / length)
                mean_non_zero_ct = np.mean(np.sum(ecfp_array, axis=1))
                print(f"computing (hv) fingerprints (length={length}, radius={radius}) took {sum_fp_time} seconds ({mean_fp_time}s/mol) with sparsity {sparsity}, with {mean_non_zero_ct} non-zero elements on average") 

                data_dict["length"].append(length)
                data_dict ["radius"].append(radius)
                data_dict["time_sum"].append(sum_fp_time)
                data_dict["time_mean"].append(mean_fp_time)
                data_dict["sparsity"].append(sparsity)
                data_dict["non_zero"].append(mean_non_zero_ct)

        df = pd.DataFrame(data_dict)
        df.to_csv("ecfp_encoding_results.csv")
    else:

        ecfp_list = [compute_fingerprint_from_smiles(x, length=args.ecfp_length, radius=args.ecfp_radius, return_time=True) for x in tqdm(smiles_df[0].values.tolist())]
        sum_fp_time = np.sum([x[1] for x in ecfp_list])
        mean_fp_time = np.mean([x[1] for x in ecfp_list])
        std_fp_time = np.std([x[1] for x in ecfp_list])
        output_path = Path(f"ecfp_{args.ecfp_length}_{args.ecfp_radius}")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        print(sum_fp_time, mean_fp_time, std_fp_time)
        np.save(output_path/Path("time.npy"), np.array([sum_fp_time, mean_fp_time]))
if __name__ == "__main__":
    # time_ecfp_encoder()
    time_direct_ecfp_encoder()
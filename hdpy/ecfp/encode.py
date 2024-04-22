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


class ECFPEncoder(Dataset):
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

        ecfp = self.compute_fingerprint_from_smiles(self.smiles_list[idx])
        if ecfp is not None:
            # start = time.time()
            hv = self.encode(ecfp)
            # end = time.time()

            return hv
        else:
            return None

    def build_item_memory(self):
        self.item_mem = torch.bernoulli(
            torch.empty(self.input_size, 2, self.D).uniform_(0, 1)
        )
        self.item_mem = torch.where(self.item_mem > 0, self.item_mem, -1).int()

    def encode(self, datapoint):
        # datapoint is just a single ECFP

        if self.item_mem is None:
            print("Build item memory before encoding")

        start = time.time()
        hv = torch.zeros(self.D, dtype=torch.int)

        for pos, value in enumerate(datapoint):
            hv += self.item_mem[pos, value]

        # binarize (why are positive values being mapped to negative values and vice versa?)
        hv = torch.where(hv > 0, hv, -1)
        hv = torch.where(hv <= 0, hv, 1)
        end = time.time()

        return hv, torch.ones(1) * (end - start)

    def compute_fingerprint_from_smiles(self, smiles):
        return compute_fingerprint_from_smiles(
            smiles=smiles, input_size=self.input_size, radius=self.radius
        )

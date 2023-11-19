################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from tkinter import E
import torch
import numpy as np
from tqdm import tqdm 
# from hdpy.hdpy.hd_model import HDModel
# import multiprocessing as mp
from torch.utils.data import Dataset
from pathlib import Path
import time

from hdpy.ecfp import compute_fingerprint_from_smiles

class ECFPEncoder(Dataset):

    def __init__(self, D:int, radius:int, input_size:int, smiles_list:list):
        super()
        # super().__init__(D=D)
        self.D = D 
        self.input_size = input_size
        self.radius = radius
        #data_path is a csv of smiles?
        self.smiles_list = smiles_list
        self.item_mem = None 
        self.name="ecfp"
        # counter to maintain running average of latency
        # self.encode_time_list = []


    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):

        #todo(derek): this could be executed multiple times
        if self.item_mem is None:
            self.build_item_memory()

        ecfp = self.compute_fingerprint_from_smiles(self.smiles_list[idx])
        if ecfp is not None:
            
            start = time.time()
            hv = self.encode(ecfp)
            end = time.time()

            # return hv, torch.ones() * end-start
            return hv
        else:
            # self.smiles_list.pop(idx)
            return None




    def build_item_memory(self):

        # self.item_mem = {"pos": torch.zeros(self.input_size, self.D).int(), "value": torch.zeros(2, self.D).int()}
        # print("building item memory")

        # for pos in range(self.input_size):
            # pos_hv = torch.bernoulli(torch.empty(self.D, dtype=torch.float32).uniform_(0,1))
            # pos_hv = torch.where(pos_hv > 0, pos_hv, -1)
            # self.item_mem["pos"][pos] = pos_hv
        
        # for value in range(2):
            # value_hv = torch.bernoulli(torch.empty(self.D, dtype=torch.float32).uniform_(0,1))
            # value_hv = torch.where(value_hv > 0, value_hv, -1)
            # self.item_mem["value"][value] = value_hv    


        # self.item_mem = torch

        # print(f"item memory formed with {len(self.item_mem['pos'].keys())} (pos) and {len(self.item_mem['value'].keys())} entries...")

        self.item_mem = torch.bernoulli(torch.empty(self.input_size, 2, self.D).uniform_(0,1))
        self.item_mem = torch.where(self.item_mem >0, self.item_mem, -1).int()


    def encode(self, datapoint):
        
        # datapoint is just a single ECFP

        if self.item_mem is None:
            print("Build item memory before encoding") 

        start = time.time()
        hv = torch.zeros(self.D, dtype=torch.int)

        for pos, value in enumerate(datapoint):

            hv += self.item_mem[pos, value]

            # if isinstance(pos,torch.Tensor):

                # hv = hv + self.item_mem["pos"][pos.data] * self.item_mem["value"][value.data]

            # else:

                # hv = hv + self.item_mem["pos"][pos] * self.item_mem["value"][value]

            # bind both item memory elements? or should I use a single 2 by n_bit matrix of values randomly chosen to associate with all possibilities?
            # hv = hv + (pos_hv * value_hv)

        # import ipdb 
        # ipdb.set_trace()

        # window_size=16
        # value_list = []
        # for i in range(1024):
            # values = torch.index_select(self.item_mem[:, i:i+window_size], 0, torch.from_numpy(datapoint).int()).int()
            # values = values.sum(dim=1)
            # value_list.append(values)
        
        # hv = torch.cat(value_list).sum(dim=0)
        # hv = self.item_mem["pos"] * values

        # hv = hv.sum(dim=0)


        # binarize (why are positive values being mapped to negative values and vice versa?)
        hv = torch.where(hv > 0, hv, -1)
        hv = torch.where(hv <= 0, hv, 1)
        end = time.time()
        # print(end-start)
        # self.encode_time_sum += end-start
        # self.n_calls += 1
        # self.encode_time_list.append(end-start)
        return hv, torch.ones(1) * (end-start)


    def compute_fingerprint_from_smiles(self, smiles):
        return compute_fingerprint_from_smiles(smiles=smiles, input_size=self.input_size, radius=self.radius)
    # def batch_encode(self, data, num_workers=16):
        # with mp.Pool(num_workers) as pool:
            # result = list(tqdm(pool.imap(self.encode, data), total=len(data), desc="encoding data in batch mode"))
        # return torch.cat(result)

################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import time
import torch
# import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
# from rdkit import Chem
from torch.utils.data import Dataset
from pathlib import Path
from hdpy.molehd.encode import tokenize_smiles
from hdpy.utils import get_random_hv, compute_fingerprint_from_smiles


class SMILESDataset(Dataset):

    def __init__(self, smiles:list, labels:list, D:int, tokenizer, ngram_order, num_workers=1, item_mem=None, device="cpu"):

        self.smiles = smiles 
        self.labels = labels
        self.D = D
        self.tokenizer = tokenizer
        self.ngram_order = ngram_order
        self.num_workers = num_workers
        self.device = device
        # self.split_df = split_df


        # import pdb
        # pdb.set_trace()


        # self.y = self.split_df[self.label_col]
        # self.smiles = smiles

        # self.smiles_train = self.split_df[self.smiles_col][self.split_df[self.split_df["split"] == "train"]["index"]]
        # self.smiles_test = self.split_df[self.smiles_col][self.split_df[self.split_df["split"] == "test"]["index"]]


        # train_encode_time = 0

        self.data_toks = tokenize_smiles(
            self.smiles, tokenizer=self.tokenizer, ngram_order=self.ngram_order, num_workers=self.num_workers,
        )
        

    # def build_item_memory(self, dataset_tokens):
        self.item_mem = item_mem
        if self.item_mem == None:
            self.item_mem = {}


        for tokens in tqdm(self.data_toks, desc="building item memory"):

            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = get_random_hv(self.D,1)
                    self.item_mem[token] = token_hv.to(self.device)

        print(f"item memory formed with {len(self.item_mem.keys())} entries.")

        
        # import pdb
        # pdb.set_trace()
        # import multiprocessing as mp
        # from functools import partial
        # with mp.Pool(16) as pool:

            # job_func = partial(tok_seq_to_hv, D=self.D, item_mem=self.item_mem)
            # self.x = np.asarray(
                    # list(tqdm(pool.imap(job_func, toks)))
                # )

        # import pdb 
        # pdb.set_trace()
        # dataset_hvs = []
    
        # for tokens in tqdm(toks, desc="computing hvs"):
            # hv = torch.zeros(self.D).int()
            # print(len(tokens))
            # for each token in the sequence, retrieve the hv corresponding to it
            # then rotate the tensor elements by the position number the token
            # occurs at in the sequence. add to (zero-initialized hv representing the 
            # for idx, token in enumerate(tokens):
                # token_hv = self.item_mem[token]
                # hv = hv + torch.roll(token_hv, idx).int()

            # binarize
            # hv = binarize(hv)
            # dataset_hvs.append(hv)
            # return hv


        # dataset_hvs = torch.cat(dataset_hvs).int()

        # import pdb
        # pdb.set_trace()

        # reduce toks to only the set of unique symbols, map these symbols torandom hypervectors (I have a function already defined for this),
        # then this dataset will just yield the hv's directly instead of what molformer and ecfp are doing

        # alternatively, we could map the symbols to some scheme that arrives at integer seqeunce representation that may be more comapact?

        # model.build_item_memory(toks)
        # train_encode_start = time.time()
        # train_dataset_hvs = model.encode_dataset(train_toks)
        # train_encode_time = time.time() - train_encode_start

        # test_encode_start = time.time()
        # test_dataset_hvs = model.encode_dataset(test_toks)
        # test_encode_time = time.time() - test_encode_start

        # train_dataset_hvs = torch.vstack(train_dataset_hvs).int()
        # test_dataset_hvs = torch.vstack(test_dataset_hvs).int()



        #todo: do this with a pool 
        # self.x_train =  np.concatenate([
                # compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                # for x in tqdm(self.smiles_train)
            # ], axis=0)

        #todo: do this with a pool 
        # self.x_test =  np.concatenate([
                # compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                # for x in tqdm(self.smiles_test)
            # ], axis=0)

        # self.y_train = self.split_df[self.label_col][self.split_df[self.split_df["split"] == "train"]["index"].values]
        # self.y_test = self.split_df[self.label_col][self.split_df[self.split_df["split"] == "test"]["index"].values]


        # self.x_train = torch.from_numpy(self.x_train).int()
        # self.x_test = torch.from_numpy(self.x_test).int()
        # self.y_train = torch.from_numpy(self.y_train).int()
        # self.y_test = torch.from_numpy(self.y_test).int()

    
    def __len__(self):

        return len(self.data_toks)

    def __getitem__(self, idx):
        # hv = torch.zeros(self.D, dtype=int)


        tokens = self.data_toks[idx]

        # for each token in the sequence, retrieve the hv corresponding to it
        # then rotate the tensor elements by the position number the token
        # occurs at in the sequence. add to (zero-initialized hv representing the 
        # for idx, token in enumerate(tokens):
            # token_hv = self.item_mem[token]
            # hv = hv + torch.roll(token_hv, idx).int()

        # binarize
        # hv = binarize(hv)
        # return hv
        return tokens, self.labels.values[idx]
        # return tokens


class ECFPDataset(Dataset):

    def __init__(self, path, smiles_col:str, label_col:str,
                  split_df:pd.DataFrame, split_type:str, ecfp_length:int, 
                  ecfp_radius:int, random_state:int, smiles:np.array, labels:list):
        super()

        # import pdb
        # pdb.set_trace()
        self.path = path
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.random_state = random_state
        self.split_df = split_df
        self.split_type = split_type
        self.ecfp_length = ecfp_length
        self.ecfp_radius = ecfp_radius
        self.labels = labels
        
        
        self.fps = np.asarray([
                compute_fingerprint_from_smiles(x, length=ecfp_length, radius=ecfp_radius)
                for x in tqdm(split_df[self.smiles_col].values.tolist())
            ])

        valid_idxs = np.array([idx for idx, x in enumerate(self.fps) if x is not None])

        self.split_df = split_df.iloc[valid_idxs]

        self.smiles = smiles

        self.smiles_train = self.smiles[self.split_df[self.split_df["split"] == "train"]["index"]]
        self.smiles_test = self.smiles[self.split_df[self.split_df["split"] == "test"]["index"]]

        #todo: do this with a pool 
        self.x_train =  np.concatenate([
                compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                for x in tqdm(self.smiles_train)
            ], axis=0)

        #todo: do this with a pool 
        self.x_test =  np.concatenate([
                compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                for x in tqdm(self.smiles_test)
            ], axis=0)

        self.y_train = self.labels[self.split_df[self.split_df["split"] == "train"]["index"].values]
        self.y_test = self.labels[self.split_df[self.split_df["split"] == "test"]["index"].values]


        self.x_train = torch.from_numpy(self.x_train).int()
        self.x_test = torch.from_numpy(self.x_test).int()
        self.y_train = torch.from_numpy(self.y_train).int()
        self.y_test = torch.from_numpy(self.y_test).int()

    def get_train_test_splits(self):

        return self.x_train, self.x_test, self.y_train, self.y_test


    def __len__(self):

        return self.x.shape[0]
    
    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]
class MolFormerDataset(Dataset):

    def __init__(self, path, split_df, smiles_col):
        super()
        self.path = path 
        self.smiles_col = smiles_col
        # self.label_col = label_col
        embeds = torch.load(path)

        x_train, x_test, y_train, y_test = [], [], [], []

        self.train_idxs, self.test_idxs = [], []
        for group, group_df in split_df.groupby("split"):
            split_idxs = group_df["index"].values

            # embed_idxs is the index_values we stored when running the molformer extraction code

            for idx in tqdm(split_idxs, desc=f"loading molformer {group} embeddings"):

                embed_idx_mask = np.equal(idx, embeds["idxs"])
                embed = embeds["embeds"][embed_idx_mask]
                label = embeds["labels"][embed_idx_mask]


                if group == "train":
                    # self.train_idxs = split_idxs
                    x_train.append(embed)
                    y_train.append(label)
                else:
                    # x_test.append(embeds["embeds"][idx])
                    x_test.append(embed)
                    y_test.append(label)

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.concatenate(y_train).reshape(-1,1)
        y_test = np.concatenate(y_test).reshape(-1,1)
        

        self.x = np.vstack([x_train, x_test])
        self.y = np.vstack([y_train, y_test]).astype(int)


        self.train_idxs = np.asarray(list(range(len(x_train))))
        self.test_idxs = np.asarray(list(range(len(x_train), len(x_train)+len(x_test))))


        # import ipdb 
        # ipdb.set_trace()

        self.smiles_train = split_df[split_df["split"] == "train"][self.smiles_col]

        self.smiles_test = split_df[split_df["split"] == "test"][self.smiles_col]
        self.smiles = pd.concat([self.smiles_train, self.smiles_test])

        self.x_train = torch.from_numpy(x_train).int()
        self.x_test = torch.from_numpy(x_test).int()
        self.y_train = torch.from_numpy(y_train).int()
        self.y_test = torch.from_numpy(y_test).int()


    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

    def get_train_test_splits(self):


        # import ipdb
        # ipdb.set_trace()
        return self.x[self.train_idxs], self.x[self.test_idxs], self.y[self.train_idxs], self.y[self.test_idxs]


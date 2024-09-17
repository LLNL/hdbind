################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import time
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from hdpy.molehd.encode import tokenize_smiles
from hdpy.utils import get_random_hv
from hdpy.ecfp import compute_fingerprint_from_smiles
# from sklearn.preprocessing import Normalizer
import multiprocessing as mp


class StreamingECFPDataset(Dataset):
    def __init__(self, smiles_list:list, labels:np.array, length:int, radius:int):
        self.smiles_list = smiles_list
        self.labels = torch.from_numpy(labels).int()
        self.length = length
        self.radius = radius

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):

        smiles = self.smiles_list[idx]
        ecfp = compute_fingerprint_from_smiles(smiles, length=self.length, radius=self.radius)

        if ecfp is None:
            print(f"compute_fingerprint_from_smiles failed for {smiles}")
        else:
            return torch.from_numpy(ecfp), self.labels[idx]

class StreamingComboDataset(Dataset):
    def __init__(self, smiles_list:list, feats:np.array, labels:np.array, length:int, radius:int):
        self.smiles_list = smiles_list
        self.labels = torch.from_numpy(labels).int()
        self.length = length
        self.radius = radius
        self.feats = feats

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):

        smiles = self.smiles_list[idx]
        ecfp = compute_fingerprint_from_smiles(smiles, length=self.length, radius=self.radius)

        # if molecule is None, we use all zeros to encode failures as no detected substructure is technically present
        if ecfp is None:
            print("ecfp is None.")
            ecfp = np.zeros(1, self.length, dtype=float)
        feat = self.feats[idx]
        # print(ecfp)
        data = np.concatenate([feat, ecfp]).astype(float)
        if ecfp is None:
            print(f"compute_fingerprint_from_smiles failed for {smiles}")
        else:
            torch_data = torch.from_numpy(data).to(torch.float)
            return torch_data, self.labels[idx]


class ComboDataset(Dataset):
    def __init__(self, smiles_list:list, feats:np.array, labels:np.array, length:int, radius:int):
        self.smiles_list = smiles_list
        self.labels = torch.from_numpy(labels).int()
        self.length = length
        self.radius = radius
        self.feats = feats

        self.ecfp_arr = np.zeros((self.labels.shape[0], self.length), dtype=int)

        for idx, smiles in tqdm(enumerate(self.smiles_list), desc="computing ecfps for combo dataset"):
            ecfp = compute_fingerprint_from_smiles(smiles, length=self.length, radius=self.radius)
            if ecfp is None:
                print("ecfp is None.")
                ecfp = np.zeros(1, self.length, dtype=int)
            self.ecfp_arr[idx, :] = ecfp

        self.ecfp_arr = torch.from_numpy(self.ecfp_arr)

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):

        # smiles = self.smiles_list[idx]
        # ecfp = compute_fingerprint_from_smiles(smiles, length=self.length, radius=self.radius)

        # if molecule is None, we use all zeros to encode failures as no detected substructure is technically present
        # if ecfp is None:
            # print("ecfp is None.")
            # ecfp = np.zeros(1, self.length)
        ecfp = self.ecfp_arr[idx, :]
        feat = self.feats[idx]
        # print(ecfp)
        data = np.concatenate([feat, ecfp]).astype(float)
        # if ecfp is None:
            # print(f"compute_fingerprint_from_smiles failed for {smiles}")
        # else:
        return torch.from_numpy(data).float(), self.labels[idx]
# '''
# todo: remove SMILESDataset that attempts to split the dataset
class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles: list,
        D: int,
        tokenizer,
        ngram_order,
        labels=None,
        num_workers=1,
        item_mem=None,
        device="cpu",
    ):
        """
        This reads in a list of smiles and labels, tokenizes the smiles,
        then yields these pairs. it also builds an item memory during
        this process which can be accessed publicly.

        """

        self.smiles = smiles
        if labels is not None:
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = labels
        self.D = D
        self.tokenizer = tokenizer
        self.ngram_order = ngram_order
        self.num_workers = num_workers
        self.device = device

        self.data_toks = tokenize_smiles(
            self.smiles,
            tokenizer=self.tokenizer,
            ngram_order=self.ngram_order,
            num_workers=self.num_workers,
        )

        self.item_mem = item_mem
        if self.item_mem == None:
            self.item_mem = {}

        self.item_mem_time = 0.0
        # for tokens in tqdm(self.data_toks, desc="building item memory"):
        for tokens in self.data_toks:
            # token_start = time.perf_counter()
            token_start = time.perf_counter()
            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = get_random_hv(self.D, 1)
                    self.item_mem[token] = token_hv.to(self.device)

            token_end = time.perf_counter()
            self.item_mem_time += token_end - token_start

        self.item_mem_time = self.item_mem_time
        # print(f"item memory formed with {len(self.item_mem.keys())} entries in {self.item_mem_time} seconds.")

    def __len__(self):
        return len(self.data_toks)

    def __getitem__(self, idx):
        if self.labels == None:
            return self.data_toks[idx]
        else:
            return self.data_toks[idx], self.labels[idx]
# '''

class RawECFPDataset(Dataset):
    def __init__(
        self, input_size: int, radius: float, D: int, num_classes: int, fp_list: list
    ):
        super()
        """
            This is just denoting the fact this dataset only yields ECFPs, not labels or etc...could probably merge this with a flag in the other case
        """
        self.input_size = input_size
        self.radius = radius
        self.D = D
        self.num_classes = num_classes
        self.ecfp_arr = torch.from_numpy(np.concatenate(fp_list)).reshape(
            -1, self.input_size
        )

    def __len__(self):
        return len(self.ecfp_arr)

    def __getitem__(self, idx):
        return self.ecfp_arr[idx]


class ECFPFromSMILESDataset(Dataset):
    def __init__(
        self,
        smiles: np.ndarray, 
        labels: np.ndarray,
        ecfp_length: int,
        ecfp_radius: int,
    ):
        super()
        # import pdb
        # pdb.set_trace()
        self.smiles = smiles
        self.labels = labels
        self.ecfp_length = ecfp_length
        self.ecfp_radius = ecfp_radius

        from functools import partial

        fp_job = partial(compute_fingerprint_from_smiles, length=ecfp_length, radius=ecfp_radius)
        with mp.Pool(int(mp.cpu_count()/2)) as pool:
            result = list(tqdm(pool.imap(fp_job, smiles), total=len(smiles)))
            pool.close()
            pool.join()

        valid_idxs = np.array([idx for idx, x in enumerate(result) if x is not None])
        self.fps = torch.from_numpy(np.asarray([x for x in result if x is not None])).int()

        self.smiles = self.smiles[valid_idxs]
        self.labels = torch.from_numpy(self.labels[valid_idxs])

        self.tensors = [self.fps, self.labels]

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, idx):
        return self.fps[idx], self.labels[idx]


class ECFPDataset(Dataset):
    def __init__(
        self,
        path,
        smiles_col: str,
        label_col: str,
        split_df: pd.DataFrame,
        split_type: str,
        ecfp_length: int,
        ecfp_radius: int,
        random_state: int,
        smiles: np.array,
        labels: list,
    ):
        super()


        self.path = path
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.random_state = random_state
        self.split_df = split_df
        self.split_type = split_type
        self.ecfp_length = ecfp_length
        self.ecfp_radius = ecfp_radius
        self.labels = labels

        self.fps = np.asarray(
            [
                compute_fingerprint_from_smiles(
                    x, length=ecfp_length, radius=ecfp_radius
                )
                for x in tqdm(split_df[self.smiles_col].values.tolist())
            ]
        )

        valid_idxs = np.array([idx for idx, x in enumerate(self.fps) if x is not None])

        self.split_df = split_df.iloc[valid_idxs]

        self.smiles = smiles

        self.smiles_train = self.smiles[
            self.split_df[self.split_df["split"] == "train"]["index"]
        ]
        self.smiles_test = self.smiles[
            self.split_df[self.split_df["split"] == "test"]["index"]
        ]

        # todo: do this with a pool?
        self.x_train = np.concatenate(
            [
                compute_fingerprint_from_smiles(
                    x, length=self.ecfp_length, radius=self.ecfp_radius
                ).reshape(1, -1)
                for x in tqdm(self.smiles_train)
            ],
            axis=0,
        )

        # todo: do this with a pool?
        self.x_test = np.concatenate(
            [
                compute_fingerprint_from_smiles(
                    x, length=self.ecfp_length, radius=self.ecfp_radius
                ).reshape(1, -1)
                for x in tqdm(self.smiles_test)
            ],
            axis=0,
        )

        self.y_train = self.labels[
            self.split_df[self.split_df["split"] == "train"]["index"].values
        ]


        self.smiles_train = self.smiles[
            self.split_df[self.split_df["split"] == "train"]["index"].values
        ]

        self.y_test = self.labels[
            self.split_df[self.split_df["split"] == "test"]["index"].values
        ]

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
                    x_train.append(embed)
                    y_train.append(label)
                else:
                    x_test.append(embed)
                    y_test.append(label)


        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)



        y_train = np.concatenate(y_train).reshape(-1, 1)
        y_test = np.concatenate(y_test).reshape(-1, 1)

        self.x = np.vstack([x_train, x_test])
        self.y = np.vstack([y_train, y_test]).astype(int)

        self.train_idxs = np.asarray(list(range(len(x_train))))
        self.test_idxs = np.asarray(
            list(range(len(x_train), len(x_train) + len(x_test)))
        )

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
        return self.x_train, self.x_test, self.y_train, self.y_test

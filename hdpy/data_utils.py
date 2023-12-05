################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
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
from sklearn.preprocessing import Normalizer



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
            token_start = time.time()
            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = get_random_hv(self.D, 1)
                    self.item_mem[token] = token_hv.to(self.device)

            token_end = time.time()
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

        # todo: do this with a pool
        self.x_train = np.concatenate(
            [
                compute_fingerprint_from_smiles(
                    x, length=self.ecfp_length, radius=self.ecfp_radius
                ).reshape(1, -1)
                for x in tqdm(self.smiles_train)
            ],
            axis=0,
        )

        # todo: do this with a pool
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

        # import pdb
        # pdb.set_trace()
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
        self.normalizer = Normalizer()
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

        # x_train = self.normalizer.transform(np.concatenate(x_train))
        # x_test = self.normalizer.transform(np.concatenate(x_test))
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
        return self.x_train, self.x_test, self.y_train, self.y_test

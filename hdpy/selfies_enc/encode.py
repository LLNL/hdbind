################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import torch
from hdpy.model import HDModel
from tqdm import tqdm
import multiprocessing as mp
import functools
import selfies as sf

constrain_dict = sf.get_semantic_constraints()
constrain_dict["N"] = 5
constrain_dict["Cl"] = 7

sf.set_semantic_constraints(constrain_dict)


class SELFIESHDEncoder(HDModel):
    def __init__(self, D):
        super(SELFIESHDEncoder, self).__init__(D=D)
        # super()
        # "D" is the dimension of the encoded representation
        self.D = D

    def build_item_memory(self, dataset_tokens):
        self.item_mem = {}

        if not isinstance(dataset_tokens[0], list):
            dataset_tokens = [dataset_tokens]

        # a little tricky but in the case of a single example need to account for that

        print("building item memory")
        for tokens in tqdm(dataset_tokens):
            # tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = torch.bernoulli(torch.empty(self.D).uniform_(0, 1))
                    token_hv = torch.where(token_hv > 0, token_hv, -1).int()
                    self.item_mem[token] = token_hv

        print(f"item memory formed with {len(self.item_mem.keys())} entries.")

    def encode(self, tokens):
        # tokens is a list of tokens, i.e. it corresponds to 1 sample

        hv = torch.zeros(self.D).int()

        for idx, token in enumerate(tokens):
            token_hv = self.item_mem[token]
            hv = hv + torch.roll(token_hv, idx).int()

        # binarize
        hv = torch.where(hv > 0, hv, -1).int()
        hv = torch.where(hv <= 0, hv, 1).int()
        return hv


def tokenize_selfies_from_smiles(smiles, charwise=False):
    # just taking the smiles as input to ease the pain of integrating this
    selfies = sf.encoder(smiles)

    # we'll store the characters in a list
    tokens = []
    if charwise:
        tokens = list(selfies)

    else:
        # the way this is written, the last element of the list will be blank as its an edge case, so remove it after splitting/replacing
        tokens = [x.replace("[", "") for x in selfies.split("]")]
        tokens = tokens[0:-1]

    return tokens


def encode_smiles_as_selfie(smiles):
    selfies = None
    try:
        selfies = sf.encoder(smiles)
    except sf.exceptions.EncoderError as e:
        print(e)
    finally:
        return selfies

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
# from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import multiprocessing as mp
# import functools
# from SmilesPE.tokenizer import SPE_Tokenizer, codecs
from hdpy.molehd import tokenize_smiles

class SMILESHDEncoder(HDModel):
    def __init__(self, D):
        super(SMILESHDEncoder, self).__init__(D=D)

        # "D" is the dimension of the encoded representation
        self.D = D
        self.name = "molehd"

    def build_item_memory(self, dataset_tokens):
        self.item_mem = {}

        if not isinstance(dataset_tokens[0], list):
            dataset_tokens = [dataset_tokens]

        print("building item memory")
        for tokens in tqdm(dataset_tokens):
            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = self.draw_random_hv()
                    self.item_mem[token] = token_hv

        print(f"item memory formed with {len(self.item_mem.keys())} entries.")

    def draw_random_hv(self):
        hv = torch.bernoulli(torch.empty(self.D).uniform_(0, 1))
        hv = torch.where(hv > 0, hv, -1).int()

        return hv

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

    def tokenize_smiles(self, smiles_list, tokenizer, ngram_order):
        return tokenize_smiles(
            smiles_list=smiles_list, tokenizer=tokenizer, ngram_order=ngram_order
        )




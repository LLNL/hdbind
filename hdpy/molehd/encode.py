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
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import multiprocessing as mp
import functools
from SmilesPE.tokenizer import SPE_Tokenizer, codecs

def foo():
    return True


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


def tokenize_smiles(smiles_list, tokenizer, ngram_order, num_workers=0):
    tokenizer_func = None
    toks = None

    tok_list = []

    spe_vob = codecs.open("/p/vast1/jones289/hd_bind_datasets/SPE_ChEMBL.txt")
    if num_workers == 0:
        spe = SPE_Tokenizer(spe_vob)

        for smiles in smiles_list:
            toks = spe.tokenize(smiles)
            toks = [x.split(" ") for x in toks]
            tok_list.extend(toks)

        return toks

    else:
        if tokenizer == "bpe":
            print("using Pre-trained SmilesPE Tokenizer")
            spe = SPE_Tokenizer(spe_vob)

            with mp.Pool(num_workers) as p:
                toks = list(
                    tqdm(
                        p.imap(spe.tokenize, smiles_list),
                        total=len(smiles_list),
                        desc="tokeninze SMILES (BPE)",
                    )
                )
                toks = [x.split(" ") for x in toks]

        else:
            if tokenizer == "atomwise":
                print("using atomwise tokenizer")
                tokenizer_func = atomwise_tokenizer

            elif tokenizer == "ngram":
                print("using kmer (n-gram) tokenizer")
                tokenizer_func = functools.partial(kmer_tokenizer, ngram=ngram_order)

            else:
                raise NotImplementedError

            with mp.Pool(num_workers) as p:
                toks = list(
                    tqdm(
                        p.imap(tokenizer_func, smiles_list),
                        total=len(smiles_list),
                        desc="tokenize SMILES ({tokenizer})",
                    )
                )

        return toks

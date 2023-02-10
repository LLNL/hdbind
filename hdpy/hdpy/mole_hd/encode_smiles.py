import torch
from hdpy.hd_model import HDModel
from tqdm import tqdm 
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import multiprocessing as mp
import functools



class SMILESHDEncoder(HDModel):

    def __init__(self, D):
        super(SMILESHDEncoder, self).__init__()

        # "D" is the dimension of the encoded representation
        self.D = D

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
                    token_hv = torch.bernoulli(torch.empty(self.D).uniform_(0,1))
                    token_hv = torch.where(token_hv > 0 , token_hv, -1).int() 
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


def tokenize_smiles(smiles, tokenizer, ngram_order):

    tokenizer_func = None
    toks = None


    if tokenizer == "bpe":
        print("using Pre-trained SmilesPE Tokenizer")
        from SmilesPE.tokenizer import SPE_Tokenizer, codecs
        spe_vob = codecs.open('/p/lustre2/jones289/hd_bind_datasets/SPE_ChEMBL.txt')
        spe = SPE_Tokenizer(spe_vob)

        with mp.Pool(mp.cpu_count()-2) as p:
            toks = list(tqdm(p.imap(spe.tokenize, smiles), total=len(smiles))) 
            toks = [x.split(' ') for x in toks]
            # import pdb 
            # pdb.set_trace()
    else:
        if tokenizer == "atomwise":
            print("using atomwise tokenizer")
            tokenizer_func = atomwise_tokenizer

        elif tokenizer == "ngram":
            print("using kmer (n-gram) tokenizer")
            tokenizer_func = functools.partial(kmer_tokenizer, ngram=ngram_order)

        else:
            raise NotImplementedError

        with mp.Pool(mp.cpu_count()-2) as p:
            toks = list(tqdm(p.imap(tokenizer_func, smiles), total=len(smiles)))

    return toks
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import multiprocessing as mp
import functools
from SmilesPE.tokenizer import SPE_Tokenizer, codecs

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
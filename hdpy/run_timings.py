################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from rdkit.Chem import DataStructs, rdmolfiles, AllChem
import time 
import pandas as pd
from pathlib import Path 
from tqdm import tqdm
from hdpy.molehd.encode import SMILESHDEncoder
from hdpy.ecfp.encode import ECFPEncoder
from hdpy.model import RPEncoder
from hdpy.data_utils import RawECFPDataset
import multiprocessing as mp
from torch.utils.data import DataLoader
from hdpy.data_utils import SMILESDataset
# import torch


def ecfp_job(smiles):

    try:

        mol_start = time.time()
        mol = rdmolfiles.MolFromSmiles(smiles)
        mol_end = time.time()

        fp_start = time.time()
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_end = time.time()

        # fp = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8), bitorder='little')
        # return (fp_vec, end - start)
        return {"ecfp": fp_vec, "smiles_to_mol_time": mol_end - mol_start, 
                "mol_to_fp_time": fp_end - fp_start}
        # return fp_vec
    
    except Exception as e:
        print(e)
        return None

# def mole_hd_job(smiles):

    # enc = SMILESHDEncoder(D=10000)

    # start = time.time()
    # toks = enc.tokenize_smiles([smiles], tokenizer="bpe", ngram_order=1)
    # enc.build_item_memory(toks)
    # hv = enc.encode(toks)
    # end = time.time()

    # return end - start 

# def ecfp_hd_job(smiles):

    # try:
        # enc = ECFPEncoder(D=10000)
        # mol = rdmolfiles.MolFromSmiles(smiles)
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # start = time.time()
        # hv = enc.encode(fp)
        # end = time.time()
    # except Exception as e:
        # print(e)
        # return None
    # return end - start

# def rp_hd_job(smiles):

    # enc = RPEncoder(D=10000, input_size=1024, num_classes=2)
    # mol = rdmolfiles.MolFromSmiles(smiles)
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    # start = time.time()
    # hv = enc.encode(fp)
    # end = time.time()

    # return end - start


def ecfp_timing(smiles_list):

    # ecfp_list = []
    # ecfp_time_list = []
    # for smiles in smiles_list:
        # x, y = ecfp_job(smiles)
        # x = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(x), dtype=np.uint8), bitorder='little')
        # ecfp_list.append(x)
        # ecfp_time_list.append(y)
    n_workers = mp.cpu_count() - 1
    print(f"using {n_workers} workers")
    with mp.Pool(n_workers) as p:
        # todo: warmup the pool first 
        _ = list(tqdm(p.imap(ecfp_job, smiles_list[:100]), total=len(smiles_list[:100]), position=1, desc="warming the pool"))        
        result_list = list(tqdm(p.imap(ecfp_job, smiles_list), total=len(smiles_list), position=2, desc="computing ECFPs"))        
    ecfp_list = [x["ecfp"] for x in result_list if x["ecfp"] is not None]
    smiles_to_mol_time_list = [x["smiles_to_mol_time"] for x in result_list if x["ecfp"] is not None]
    ecfp_time_list = [x["mol_to_fp_time"] for x in result_list if x["ecfp"] is not None]

    return ecfp_list, ecfp_time_list, smiles_to_mol_time_list


from hdpy.data_utils import SMILESDataset
from hdpy.model import TokenEncoder
def smiles_token_hdc(smiles_list, tokenizer, ngram_order, D:int, batch_size:int, num_workers):

    dataset = SMILESDataset(smiles=smiles_list, tokenizer=tokenizer, ngram_order=ngram_order, D=D)

    item_mem_time = dataset.item_mem_time

    tok_encoder = TokenEncoder(D=D, num_classes=2, item_mem=dataset.item_mem)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn=lambda data: [x for x in data])

    tok_encode_time_list = []

    for batch in dataloader:
        
        tok_encode_start = time.time()
        tok_encoder.encode_batch(batch)
        tok_encode_end = time.time()

        tok_encode_time_list.append(tok_encode_end - tok_encode_start)


# def hdc_timing(smiles_list, fp_list, method:str, input_size:int, radius:int, batch_size:int):

    # enc = None
    # import ipdb 
    # ipdb.set_trace()

    # if method == "rphd":
        # rp_enc = RPEncoder(input_size=input_size, D=10000, num_classes=2).to(0)
        # enc = RawECFPDataset(D=10000, input_size=input_size, radius=radius, num_classes=2,
                                # fp_list=fp_list)
        
        # dataloader = DataLoader(enc, num_workers=0, batch_size=batch_size)

        # import ipdb 
        # ipdb.set_trace()
        # batch_time_list = []
        # for batch in tqdm(dataloader):
            # start = time.time()
            # _ = rp_enc.encode(batch.to(0))
            # end = time.time()
            # batch_time_list.append((end-start)/batch_size)

        # return np.mean(batch_time_list)
    # else:

        # if method == "ecfphd":
            # enc = ECFPEncoder(D=10000, input_size=input_size, radius=radius, fp_list=fp_list)
        # if method == "molehd":
            # enc = SMILESHDEncoder(D=10000, smiles_list=smiles_list)
            
        # dataloader = DataLoader(enc, num_workers=0, batch_size=batch_size)
        # import ipdb
        # ipdb.set_trace()
        # timing_list = []
        # for batch in tqdm(dataloader):
            # _, timings = batch 
            # timing_list.append(timings)

        # return timing_list


        




def lit_pcba_main(input_size:int, radius:int):

    lit_pcba_dir = Path("/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/")



    df_list = []
    for path in tqdm(list(lit_pcba_dir.glob("*/")),desc=f"processing target", position=0):
    #  ADRB2/smiles_train.csv  

        train_path = path / Path("smiles_train.csv")
        test_path = path / Path("smiles_test.csv")


        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        df = pd.concat([train_df, test_df], axis=0).sample(frac=0.25)

        target_smiles = df[0].values.tolist()
        _, ecfp_timing_list, smiles_to_mol_time_list = ecfp_timing(smiles_list=target_smiles)


        smiles_token_hdc(smiles_list=target_smiles, tokenizer="bpe", ngram_order=0,
                         D=10000, batch_size=128, num_workers=16)


        df_list.append(pd.DataFrame(
            {"target": [path.name] * len(ecfp_timing_list), "ecfp_time": ecfp_timing_list, "smiles_to_mol_time": smiles_to_mol_time_list}
        ))


        # timing_df = compute_timing(df[0].values.tolist())
        # timing_df_list.append(timing_df)
        # print(path, mean_time)
    # return pd.DataFrame({"rphd_time": rphd_timing_list, "molehd_time": molehd_timing_list})
    # return pd.DataFrame({"ecfp_time": ecfp_timing_list})

    df = pd.concat(df_list).reset_index()

    return df
if __name__ == "__main__":
    import numpy as np


    lit_pcba_df = lit_pcba_main(1024, 2)


    # print(np.mean(time_list))
    # print(len(time_list))
    #print(df.shape[0])

    import pandas as pd

    # df = pd.DataFrame({"time_mean": [np.mean(time_list)], "n": [len(time_list)], "time_std": [np.std(time_list)]})
    lit_pcba_df.to_csv("timings.csv", header=True, index=False)
    print()
    print(lit_pcba_df.mean())
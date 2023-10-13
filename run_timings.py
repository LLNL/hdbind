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
from hdpy.hdpy.mole_hd.encode import SMILESHDEncoder
from hdpy.hdpy.ecfp_hd.encode import ECFPEncoder
from hdpy.hdpy.baseline_hd.classification_modules import ECFPDataset, RPEncoder
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch
def ecfp_job(smiles):

    try:

        mol = rdmolfiles.MolFromSmiles(smiles)

        start = time.time()

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        end = time.time()

        # fp = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8), bitorder='little')
        return (fp_vec, end - start)
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

    ecfp_list = []
    ecfp_time_list = []
    for smiles in smiles_list:
        x, y = ecfp_job(smiles)
        x = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(x), dtype=np.uint8), bitorder='little')
        ecfp_list.append(x)
        ecfp_time_list.append(y)
    # with mp.Pool(64) as p:

        # todo: warmup the pool first 

        # ecfp_time_list = list(tqdm(p.imap(ecfp_job, smiles_list[:100]), total=len(smiles_list[:100]), position=0, desc="computing ecfp"))        


        # result_list = list(tqdm(p.imap(ecfp_job, smiles_list), total=len(smiles_list), position=0, desc="computing ecfp"))        
    # ecfp_list = [ for x in result_list if x[0] is not None]
    # ecfp_time_list = [x[1] for x in result_list]

    return ecfp_list, ecfp_time_list


def hdc_timing(smiles_list, fp_list, method:str, input_size:int, radius:int, batch_size:int):

    enc = None
    # import ipdb 
    # ipdb.set_trace()

    if method == "rphd":
        rp_enc = RPEncoder(input_size=input_size, D=10000, num_classes=2).to(0)
        enc = ECFPDataset(D=10000, input_size=input_size, radius=radius, num_classes=2,
                                fp_list=fp_list)
        
        dataloader = DataLoader(enc, num_workers=0, batch_size=batch_size)

        # import ipdb 
        # ipdb.set_trace()
        batch_time_list = []
        for batch in tqdm(dataloader):
            start = time.time()
            _ = rp_enc.encode(batch.to(0))
            end = time.time()
            batch_time_list.append((end-start)/batch_size)

        return np.mean(batch_time_list)
    else:

        if method == "ecfphd":
            enc = ECFPEncoder(D=10000, input_size=input_size, radius=radius, fp_list=fp_list)
        if method == "molehd":
            enc = SMILESHDEncoder(D=10000, smiles_list=smiles_list)
            
        dataloader = DataLoader(enc, num_workers=0, batch_size=batch_size)
        # import ipdb
        # ipdb.set_trace()
        timing_list = []
        for batch in tqdm(dataloader):
            _, timings = batch 
            timing_list.append(timings)

        return timing_list


        




def lit_pcba_main(input_size:int, radius:int):

    lit_pcba_dir = Path("/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/")



    rphd_timing_list = []
    molehd_timing_list = []
    for path in tqdm(list(lit_pcba_dir.glob("*/")),desc=f"processing target", position=1):
    #  ADRB2/smiles_train.csv  

        train_path = path / Path("smiles_train.csv")
        test_path = path / Path("smiles_test.csv")


        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        df = pd.concat([train_df, test_df], axis=0).sample(frac=0.25)

        target_smiles = df[0].values.tolist()
        ecfp_list, ecfp_timing_list = ecfp_timing(smiles_list=target_smiles)

        rphd_mean_time = hdc_timing(smiles_list=target_smiles, fp_list=ecfp_list,
                            method="rphd", input_size=input_size, radius=2,
                            batch_size=512)
        rphd_timing_list.append(rphd_mean_time)

        molehd_mean_time = hdc_timing(smiles_list=target_smiles, fp_list=ecfp_list,
                            method="molehd", input_size=input_size, radius=2,
                            batch_size=512)
        molehd_timing_list.append(molehd_mean_time)       


        # timing_df = compute_timing(df[0].values.tolist())
        # timing_df_list.append(timing_df)
        # print(path, mean_time)
    return pd.DataFrame({"rphd_time": rphd_timing_list, "molehd_time": molehd_timing_list})

if __name__ == "__main__":
    import numpy as np


    lit_pcba_df = lit_pcba_main(1024, 2)


    # print(np.mean(time_list))
    # print(len(time_list))
    #print(df.shape[0])

    import pandas as pd

    # df = pd.DataFrame({"time_mean": [np.mean(time_list)], "n": [len(time_list)], "time_std": [np.std(time_list)]})
    lit_pcba_df.to_csv("morgan_fp_lit_pcba_timing.csv", header=True, index=False)

    print(lit_pcba_df)
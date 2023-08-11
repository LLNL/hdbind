from rdkit.Chem import DataStructs, rdmolfiles, AllChem
import time 
import pandas as pd



from pathlib import Path 


lit_pcba_dir = Path("/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/")

from tqdm import tqdm 
for path in tqdm(list(lit_pcba_dir.glob("*/"))):
#  ADRB2/smiles_train.csv  

    train_path = path / Path("smiles_train.csv")
    test_path = path / Path("smiles_test.csv")


    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    df = pd.concat([train_df, test_df], axis=0)


    time_list = []
    for smiles in df[0].values.tolist():

        mol = rdmolfiles.MolFromSmiles(smiles)

        start = time.time()

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        end = time.time()

        time_list.append(end - start)

import numpy as np

print(np.mean(time_list))
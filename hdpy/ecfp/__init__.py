################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import time
import numpy as np
from rdkit.Chem import DataStructs
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def compute_fingerprint_from_smiles(smiles, length: int, radius: int, return_time=False):
    try:
        start = time.perf_counter()
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)
        # mol = rdmolfiles.MolFromSmiles(smiles, sanitize=False)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=length)

        fp = np.unpackbits(
            np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8),
            bitorder="little",
        )
        end=time.perf_counter()
        # print(fp, fp_vec)
        if return_time:
            return fp, end - start
        else:
         return fp

    except ValueError as e:
        print(e)
        return None

    except Exception as e:
        print(e)
        return None




def compute_fingerprint_from_smiles_parallel(smiles, length, radius, n_workers):

    with mp.Pool(n_workers) as pool:
        
        job_func = partial(compute_fingerprint_from_smiles, length=length, radius=radius)
        result = list(tqdm(pool.imap(job_func, smiles), total=len(smiles)))

    return result
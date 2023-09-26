import numpy as np
from rdkit.Chem import DataStructs
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem




def compute_fingerprint_from_smiles(smiles, length:int, radius:int):
    try:
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)
        # mol = rdmolfiles.MolFromSmiles(smiles, sanitize=False)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=length)


        fp = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8), bitorder='little')
        # print(fp, fp_vec)
        return fp
    
    except Exception as e:
        print(e)
        return None


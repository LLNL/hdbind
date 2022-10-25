from tkinter import E
import torch
import numpy as np
from tqdm import tqdm 
from hdpy.hd_model import HDModel
from rdkit.Chem import DataStructs
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem



class ECFPEncoder(HDModel):


    def __init__(self, D):
        super(ECFPEncoder, self).__init__()

        self.D = D 

    def build_item_memory(self, n_bits:int):

        self.item_mem = {"pos": {}, "value": {}}


        print("building item memory")

        for pos in tqdm(range(n_bits)):
            pos_hv = torch.bernoulli(torch.empty(self.D, dtype=torch.float32).uniform_(0,1))
            pos_hv = torch.where(pos_hv > 0, pos_hv, -1)
            self.item_mem["pos"][pos] = pos_hv
        
        for value in range(2):
            value_hv = torch.bernoulli(torch.empty(self.D, dtype=torch.float32).uniform_(0,1))
            value_hv = torch.where(value_hv > 0, value_hv, -1)
            self.item_mem["value"][value] = value_hv    

        
        print(f"item memory formed with {len(self.item_mem['pos'].keys())} (pos) and {len(self.item_mem['value'].keys())} entries...")


    def encode(self, datapoint):
        
        # datapoint is just a single ECFP

        # import pdb
        # pdb.set_trace()
        hv = torch.zeros(self.D, dtype=torch.float32)

        for pos, value in enumerate(datapoint):

            if isinstance(pos,torch.Tensor):

                hv = hv + self.item_mem["pos"][pos.data] * self.item_mem["value"][value.data]

            else:

                hv = hv + self.item_mem["pos"][pos] * self.item_mem["value"][value]

            # bind both item memory elements? or should I use a single 2 by n_bit matrix of values randomly chosen to associate with all possibilities?
            # hv = hv + (pos_hv * value_hv)

        # binarize
        hv = torch.where(hv > 0, hv, -1)
        hv = torch.where(hv <= 0, hv, 1)

        return hv





def compute_fingerprint_from_smiles(smiles):
    try:
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


        fp = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8), bitorder='little')
        return fp
    except Exception as e:
        print(e)
        return None


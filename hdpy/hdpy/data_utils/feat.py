import argparse
import numpy as np
import pandas as pd
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem
import multiprocessing as mp
from tqdm import tqdm 
from pathlib import Path
import deepchem as dc
from rdkit.Chem import DataStructs
from rdkit.rdBase import BlockLogs

block = BlockLogs()

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--smiles-col', required=True)
    parser.add_argument('--label-col', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--feat-type', choices=["ecfp", "smiles_to_seq", "smiles_to_image",
                                    "coul_matrix", "mordred", "maacs", "rdkit"])
    parser.add_argument('--no-subset', action='store_true', help="flag that indicates no train/val/test splits are used")
    parser.add_argument('--dry-run', action="store_true")
    args = parser.parse_args()

    return args 


def compute_fingerprint(smiles_row):
    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    try:
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


        fp = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8), bitorder='little')
        return fp
    except Exception as e:
        print(e)
        return None


def compute_smiles_seq_vector(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat_seq = dc.feat.SmilesToSeq(char_to_idx=char_to_idx).featurize(smiles)

    return feat_seq


def compute_smiles_to_image(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.SmilesToImage(img_size=80, img_spec='std').featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_mordred(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.MordredDescriptors(ignore_3D=False).featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_coulmatrix(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    max_atoms=100
    feat = dc.feat.CoulombMatrix(max_atoms=max_atoms).featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_mol2vec(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.Mol2VecFingerprint().featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_maccs(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.MACCSKeysFingerprint().featurize(smiles)
    return feat.reshape(-1)

def compute_smiles_to_rdkit(smiles_row):

    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.RDKitDescriptors().featurize(smiles)
    return feat.reshape(-1)


def compute_char_to_idx(smiles_list):
    _char_to_idx = {}
    _char_idx = 0

    # can do this in parallel
    for smiles in smiles_list:
        for char in smiles:

            if char in _char_to_idx.keys():
                continue
            else:
                _char_to_idx[char] = _char_idx
                _char_idx += 1

    _char_to_idx['<unk>'] = _char_idx
    return _char_to_idx



def main():

    labels = df[args.label_col].values.reshape(-1, 1) 

    # import pdb
    # pdb.set_trace()    
    with mp.Pool(mp.cpu_count() - 1) as pool:

    #    result_list = list(tqdm(pool.imap(job_func, smiles_list), total=len(smiles_list)))

        result_list = list(tqdm(pool.imap(job_func, df.iterrows()), total=len(df)))

    data = np.asarray(result_list)
    data = data.squeeze()

    if not args.dry_run:
        output_path =  Path(f"{args.output_dir}")
        output_path.mkdir(exist_ok=True, parents=True)


        if not args.no_subset:

            train_data = np.concatenate([data[train_mask, :], labels[train_mask]], axis=1)
            val_data = np.concatenate([data[val_mask, :], labels[val_mask]], axis=1)
            test_data = np.concatenate([data[test_mask, :], labels[test_mask]], axis=1)

            np.save(output_path / "train.npy", train_data)
            np.save(output_path / "val.npy", val_data)
            np.save(output_path / "test.npy", test_data)
        
        else:

            data = np.concatenate([data,labels], axis=1)
            np.save(output_path / "data.npy", data)
        

if __name__ == "__main__":

    args = get_args()
    print(args)

    df = pd.read_csv(args.input_path)
    smiles_list = df[args.smiles_col].values.tolist()


    char_to_idx = None
    job_func = None
    if args.feat_type.lower() == "ecfp":
        job_func = compute_fingerprint

    elif args.feat_type.lower() == "smiles_to_seq":
        char_to_idx = compute_char_to_idx(smiles_list)
        job_func = compute_smiles_seq_vector
    
    elif args.feat_type.lower() == "smiles_to_image":
        job_func = compute_smiles_to_image

    elif args.feat_type.lower() == "coul_matrix":
        job_func = compute_smiles_to_coulmatrix
    
    elif args.feat_type.lower() == "mordred":
        job_func = compute_smiles_to_mordred

    elif args.feat_type.lower() == "maacs":
        job_func = compute_smiles_to_maccs
    
    elif args.feat_type.lower() == "rdkit":
        job_func = compute_smiles_to_rdkit


    main()


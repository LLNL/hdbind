################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
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
from sklearn.model_selection import train_test_split

block = BlockLogs()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--input-path-list", nargs="+")
    parser.add_argument("--smiles-col", required=True)
    parser.add_argument("--label-col")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feat-type",
        choices=[
            "ecfp",
            "smiles_to_seq",
            "smiles_to_image",
            "coul_matrix",
            "mordred",
            "maacs",
            "rdkit",
            "mol2vec",
        ],
    )
    parser.add_argument(
        "--no-subset",
        action="store_true",
        help="flag that indicates no train/val/test splits are used",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--invert-labels",
        action="store_true",
        help="pass this flag to flip the polarity of bind/no-bind labels",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers to use to featurize the data",
        default=int(mp.cpu_count() / 2),
    )
    args = parser.parse_args()

    return args


def compute_fingerprint(smiles):
    # _, smiles_row = smiles_row
    # smiles = smiles_row[args.smiles_col]
    try:
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        fp = np.unpackbits(
            np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8),
            bitorder="little",
        )
        return fp
    except Exception as e:
        print(e)
        return None


def compute_fingerprint_from_smiles(smiles):
    try:
        mol = rdmolfiles.MolFromSmiles(smiles, sanitize=True)

        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        fp = np.unpackbits(
            np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8),
            bitorder="little",
        )
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
    feat = dc.feat.SmilesToImage(img_size=80, img_spec="std").featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_mordred(smiles_row):
    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.MordredDescriptors(ignore_3D=False).featurize(smiles)
    return feat.reshape(-1)


def compute_smiles_to_coulmatrix(smiles_row):
    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    max_atoms = 100
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


def compute_smiles_to_mol2vec(smiles_row):
    _, smiles_row = smiles_row
    smiles = smiles_row[args.smiles_col]
    feat = dc.feat.Mol2VecFingerprint().featurize(smiles)

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

    _char_to_idx["<unk>"] = _char_idx
    return _char_to_idx


def main(df, pool, target):
    labels = None
    if args.label_col:
        labels = df[args.label_col].values.reshape(-1, 1)

    if args.invert_labels:
        labels = 1 - labels

    # import pdb
    # pdb.set_trace()
    # with mp.Pool(args.num_workers) as pool:

    #    result_list = list(tqdm(pool.imap(job_func, smiles_list), total=len(smiles_list)))
    result_list = list(tqdm(pool.imap(job_func, df[0].values.tolist()), total=len(df)))

    data = np.asarray(result_list)
    data = data.squeeze()

    if not args.dry_run:
        output_path = Path(f"{args.output_dir}/{target}/{args.feat_type}")
        output_path.mkdir(exist_ok=True, parents=True)

        if not args.no_subset:
            # TODO: use precomputed splits or stratify the split

            x_train, x_test, y_train, y_test = train_test_split(
                data, labels, test_size=0.2, stratify=labels, random_state=0
            )

            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.2, stratify=y_train, random_state=0
            )

            train_data = np.concatenate([x_train, y_train], axis=1)
            val_data = np.concatenate([x_val, y_val], axis=1)
            test_data = np.concatenate([x_test, y_test], axis=1)

            np.save(output_path / "train.npy", train_data)
            np.save(output_path / "val.npy", val_data)
            np.save(output_path / "test.npy", test_data)

        elif args.label_col:
            data = np.concatenate([data, labels], axis=1)
            np.save(output_path / "data.npy", data)

        else:
            np.save(output_path / "data.npy", data)


if __name__ == "__main__":
    args = get_args()
    print(args)

    # this is specific for dude...could generalize this with an argument for dataset type then use a series of if/else

    pool = mp.Pool(args.num_workers)
    for path in tqdm(args.input_path_list):
        tqdm.write(path)
        input_path = Path(path)
        target = input_path.stem.split("_")[0]

        path_df = pd.read_csv(input_path, header=None, delim_whitespace=True)

        # smiles_list = path_df[args.smiles_col].values.tolist()

        smiles_list = path_df[0].values.tolist()
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

        elif args.feat_type.lower() == "mol2vec":
            job_func = compute_smiles_to_mol2vec

        main(df=path_df, pool=pool, target=target)

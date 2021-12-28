import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import struct
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem

from hdpy.pickle_to_choir import writeDataSetForChoirSIM


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    help="path to input pdbbind directory, script will infer the structure as having a top level with pdbid and in each "
    + "subdirectory there will be a pdbid_ligand.mol2 file",
)
parser.add_argument(
    "--output-prefix", help="prefix to append to output pickle", required=True
)
parser.add_argument(
    "--output-dir", help="path to the output file that will be generated", required=True
)
parser.add_argument(
    "--n-cores", type=int, help="number of cores to use for processing", default=1
)
parser.add_argument(
    "--metadata",
    help="path to metadata file that contains label information",
    required=True,
)
parser.add_argument(
    "--pdbid-list",
    help="path to file containing list of pdbids to use for processing. if left unspecified, will process the whole input directory.",
    default=None,
)
args = parser.parse_args()
import multiprocessing as mp


def compute_fingerprint(smiles_path):

    # TODO: sanitize results in most of the molecules raising errors, could dig deeper into this https://www.rdkit.org/docs/RDKit_Book.html

    # infering the position of the pdbid from the position of the leaf of the path
    pdbid = str(smiles_path).split("/")[-2]
    try:
        mol = rdmolfiles.MolFromMol2File(str(smiles_path), sanitize=False)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        smiles = Chem.MolToSmiles(mol)
    except:
        return
    return pdbid, fp, smiles


def process_main(mol_path_list, bind_map):

    if args.n_cores == 1:
        fp_list = []
        for mol_path in mol_path_list:
            fp_list.append(compute_fingerprint(mol_path))
    else:
        with mp.Pool(mp.cpu_count() - 1) as pool:

            fp_list = list(
                tqdm(
                    pool.imap(compute_fingerprint, mol_path_list),
                    total=len(mol_path_list),
                )
            )

    # filter the None out of the fp_list
    fp_list = [x for x in fp_list if x is not None]

    pdbid_list = [x[0] for x in fp_list]
    fp_array = np.asarray([x[1] for x in fp_list])
    smiles_list = [x[2] for x in fp_list]

    bind_list = [bind_map[x] for x in pdbid_list]
    bind_array = np.asarray(bind_list)

    data_tup = (fp_array, bind_array, pdbid_list, smiles_list)

    return data_tup


def main():

    p = Path(args.input_dir)
    mol_path_list = [str(x) for x in list(p.glob("**/*ligand.mol2"))]
    mol_path_df = pd.DataFrame({"file": mol_path_list})
    mol_path_df["pdbid"] = mol_path_df["file"].apply(lambda x: x.split("/")[-2])

    metadata_df = pd.read_csv(args.metadata)
    metadata_df = pd.merge(mol_path_df, metadata_df, on="pdbid")

    if args.pdbid_list:
        pdbid_list = pd.read_csv(args.pdbid_list, header=None)[0].values.tolist()
        print(f"{len(pdbid_list)} pdbids found in args.pdbid_list")

        metadata_df = metadata_df[metadata_df["pdbid"].apply(lambda x: x in pdbid_list)]
        print(f"{metadata_df.shape[0]} pdbids remaining after filter")

    bind_map = {
        value[0]: value[1] for key, value in metadata_df[["pdbid", "bind"]].iterrows()
    }

    fps, labels, pdbids, smiles = process_main(metadata_df["file"].values, bind_map)

    data_tup = (fps, labels)

    output_dir = Path(args.output_dir)
    # create output dir if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir()

    writeDataSetForChoirSIM(
        ds=data_tup, filename=f"{output_dir}/{args.output_prefix}.choir_dat"
    )

    out_df = pd.DataFrame({"pdbid": pdbids})
    out_df = pd.concat([out_df, pd.DataFrame({"smiles": smiles})], axis=1)
    out_df = pd.concat([out_df, pd.DataFrame({"label": labels})], axis=1)
    out_df = pd.concat([out_df, pd.DataFrame(fps)], axis=1)

    out_df.to_csv(f"{output_dir}/{args.output_prefix}_pdbid_list.csv")


if __name__ == "__main__":
    main()

''' utility script to sanitize input set of smiles for rdkit processing '''


import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to input csv with smiles', 
default="/g/g13/jones289/workspace/hd-cuda-master/datasets/pdbbind/metadata/pdbbind_2019_metadata_full_with_smiles.csv")
parser.add_argument('--smiles-col')
parser.add_argument('-o', help='path to output csv file with processed smiles', required=True)
args = parser.parse_args()



import rdkit 
from rdkit import Chem 
from rdkit.Chem import rdmolfiles, rdmolops

# lifted from https://github.com/ATOMScience-org/AMPL/blob/e218870f352aabcf4a134d3928300c9a72d7c605/atomsci/ddm/utils/struct_utils.py
def get_rdkit_smiles(orig_smiles, useIsomericSmiles=True):
    """
    Given a SMILES string, regenerate a "canonical" SMILES string for the same molecule
    using the implementation in RDKit.
    Args:
        orig_smiles (str): SMILES string to canonicalize.
        useIsomericSmiles (bool): Whether to retain stereochemistry information in the generated string.
    Returns:
        str: Canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(orig_smiles)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=useIsomericSmiles)


def process_smiles(smiles):
	# mol = rdmolfiles.MolFromSmiles(smiles, sanitize=False)
	# mol = rdmolops.RemoveHs(mol, sanitize=False)
	# rdkit_smiles = rdmolfiles.MolToSmiles(mol)
	rdkit_smiles = get_rdkit_smiles(smiles)
	return rdkit_smiles

def main():

	import pandas as pd 

	df = pd.read_csv(args.i)

	from tqdm import tqdm 
	import multiprocessing as mp 


	with mp.Pool(mp.cpu_count() - 1) as p:
		result = list(tqdm(p.imap(process_smiles, df[args.smiles_col]), total=len(df)))

	df['rdkit_smiles'] = result

	# import pdb
	# pdb.set_trace()

	df.to_csv(args.o, index=False)

if __name__ == '__main__':

	main()

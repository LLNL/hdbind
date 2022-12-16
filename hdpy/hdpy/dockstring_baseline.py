from dockstring import load_target


target = load_target("BACE1")


import pandas as pd


df = pd.read_csv("/g/g13/jones289/workspace/hd-cuda-master/datasets/dude/dude_smiles/bace1_gbsa_smiles.csv")


from tqdm import tqdm 
for idx, row in tqdm(df.iterrows(), total=len(df)):

	smiles = row['smiles']

	try:

		score, info = target.dock(smiles)
	except Exception as e:
		print(e)
		continue

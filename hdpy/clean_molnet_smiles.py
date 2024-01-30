################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################

from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--smiles-col")
args = parser.parse_args()

df = pd.read_csv(args.input)

result_list = []
for smiles in df[args.smiles_col]:
    try:
        MolFromSmiles(smiles)
        MurckoScaffoldSmiles(smiles)
        result_list.append(True)
    except Exception as e:
        print(e)
        result_list.append(False)

df["valid_rdkit"] = result_list

from pathlib import Path

output_path = Path(args.output)
if not output_path.exists():
    output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path)

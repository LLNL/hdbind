################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import sys
from rdkit import Chem


def converter(file_name):
    sppl = Chem.SDMolSupplier(file_name)
    sppl = [Chem.MolFromMol2File(file_name)]
    outname = file_name.replace(".sdf", ".txt")
    out_file = open(outname, "w")
    for mol in sppl:
        if mol is not None:  # some compounds cannot be loaded.
            smi = Chem.MolToSmiles(mol, sanitize=False)
            name = mol.GetProp("_Name")
            out_file.write(f"{smi}\t{name}\n")
    out_file.close()


if __name__ == "__main__":
    converter(sys.argv[1])

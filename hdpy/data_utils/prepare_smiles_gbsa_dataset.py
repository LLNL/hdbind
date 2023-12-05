################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import os
from pathlib import Path


p = Path("/usr/workspace/atom/gbsa_modeling/dude_smiles/")


for path in p.glob("*_gbsa_smiles_with_base_rdkit_smiles.csv"):
    name = path.name.split("_")[0]
    print(f"{name}:\t{path}")
    output_dir = (
        f"/usr/workspace/atom/gbsa_modeling/dude_smiles/deepchem_feats_labeled_by_gbsa"
    )
    # for feat in ["ecfp", "smiles_to_seq", "smiles_to_image", "coul_matrix", "mordred", "maacs", "rdkit"]:
    for feat in [
        "ecfp",
        "smiles_to_seq",
        "smiles_to_image",
        "mordred",
        "maacs",
        "rdkit",
    ]:
        cmd = f"python feat.py --input-path {path} --output-dir {output_dir}/{name}/{feat} --feat-type {feat} --smiles-col base_rdkit_smiles --label-col best_gbsa_score --no-subset"
        os.system(cmd)

################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import pickle
import numpy
import struct
import numpy as np
from pathlib import Path
import deepchem as dc
from deepchem.molnet import load_hiv
from hdpy.data_utils import ECFPFromSMILESDataset
from tqdm import tqdm

def writeDataSetForChoirSIM(ds, filename):
    # import pdb
    # pdb.set_trace()
    X, y = ds
    f = open(filename, "wb")
    nFeatures = len(X[0])
    nClasses = len(set(y))

    f.write(struct.pack("i", nFeatures))
    f.write(struct.pack("i", nClasses))
    for V, l in zip(X, y):
        for v in V:
            f.write(struct.pack("f", v))
        f.write(struct.pack("i", l))


def numpy_to_choirdat(X, y, filename):

    f = open(filename, "wb")
    nFeatures = X.shape[1]
    nClasses = np.unique(y).shape[0]

    # import pdb 
    # pdb.set_trace()

    f.write(struct.pack("i", nFeatures))
    f.write(struct.pack("i", nClasses))
    for V, l in tqdm(list(zip(X, y))):
        for v in V:
            f.write(struct.pack("f", v))
        f.write(struct.pack("i", l))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["hiv"])
    parser.add_argument("--output-dir", default="hdbind_choirdat", help="path to output directory to store dataset files")

    args = parser.parse_args()


    if args.dataset == "hiv":
        
        print("featurizing MoleculeNet hiv dataset for choirdat format..")
        smiles_featurizer = dc.feat.DummyFeaturizer()
        dataset = load_hiv(splitter="scaffold", featurizer=smiles_featurizer)
        target_list = dataset[0]

        # use something besides train_dataset/test_dataset?
        train_dataset = dataset[1][0]
        test_dataset = dataset[1][1]

        smiles_train = train_dataset.X
        y_train = train_dataset.y

        smiles_test = test_dataset.X
        y_test = test_dataset.y

        train_dataset = ECFPFromSMILESDataset(smiles=smiles_train, 
                                        labels=y_train, 
                                        ecfp_length=1024,
                                        ecfp_radius=1)
            
        test_dataset = ECFPFromSMILESDataset(smiles=smiles_test,
                                    labels=y_test,
                                    ecfp_length=1024,
                                    ecfp_radius=1)


        ecfp_train = train_dataset.fps.numpy()

        ecfp_test = test_dataset.fps.numpy()

    output_dir_path = Path(f"{args.output_dir}")
    if not output_dir_path.exists():
        print(f"creating {output_dir_path}..")
        output_dir_path.mkdir(parents=True)
    train_path = output_dir_path / Path(f"{args.dataset}_train.choirdat")
    test_path = output_dir_path / Path(f"{args.dataset}_test.choirdat")
    numpy_to_choirdat(X=ecfp_train.astype(np.int8), y=y_train.flatten().astype(np.int8), filename=train_path)
    numpy_to_choirdat(X=ecfp_test.astype(np.int8), y=y_test.flatten().astype(np.int8), filename=test_path)

    print(f"done. train: {train_path}\ttest: {test_path}")

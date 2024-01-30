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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["dude", "lit-pcba", "lit-pcba-ave"])

    args = parser.parse_args()

    if args.dataset == "dude":
        dude_root_p = Path(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/dude/deepchem_feats"
        )

        for path in dude_root_p.glob("*/ecfp"):
            print(path)
            train_p = path / Path("train.npy")
            test_p = path / Path("test.npy")

            x_train = np.load(train_p)
            x_test = np.load(test_p)

            x_train, y_train = x_train[:, :-1], x_train[:, -1]
            x_test, y_test = x_test[:, :-1], x_test[:, -1]

            # DUDE data labels are messed up, inverted
            y_train = 1 - y_train
            y_test = 1 - y_test

            print(len(x_train))
            print(len(y_train))
            print(len(x_test))
            print(len(y_test))

            writeDataSetForChoirSIM(
                (x_train, y_train), train_p.with_name("train.choir_dat")
            )
            writeDataSetForChoirSIM(
                (x_test, y_test), test_p.with_name("test.choir_dat")
            )

    elif args.dataset == "lit-pcba":
        lit_pcba_root_p = Path(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data"
        )

        for path in lit_pcba_root_p.glob("*"):
            print(path)

            train_p = path / Path("random_split_train.npy")
            test_p = path / Path("random_split_test.npy")

            x_train = np.load(train_p)
            x_test = np.load(test_p)

            x_train, y_train = x_train[:, :-1], x_train[:, -1]
            x_test, y_test = x_test[:, :-1], x_test[:, -1]

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            train_choir_p = train_p.with_name("random_split_train.choir_dat")
            test_choir_p = test_p.with_name("random_split_test.choir_dat")

            print(train_choir_p, test_choir_p)

            writeDataSetForChoirSIM((x_train, y_train), train_choir_p)
            writeDataSetForChoirSIM((x_test, y_test), test_choir_p)

    elif args.dataset == "lit-pcba-ave":
        lit_pcba_root_p = Path(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/AVE_unbiased"
        )

        for path in lit_pcba_root_p.glob("*"):
            print(path)

            train_p = path / Path("ecfp_train.npy")
            test_p = path / Path("ecfp_test.npy")

            x_train = np.load(train_p)
            x_test = np.load(test_p)

            x_train, y_train = x_train[:, :-1], x_train[:, -1]
            x_test, y_test = x_test[:, :-1], x_test[:, -1]

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            train_choir_p = train_p.with_name("ecfp_train.choir_dat")
            test_choir_p = test_p.with_name("ecfp_test.choir_dat")

            print(train_choir_p, test_choir_p)

            writeDataSetForChoirSIM((x_train, y_train), train_choir_p)
            writeDataSetForChoirSIM((x_test, y_test), test_choir_p)

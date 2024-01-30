################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################

import argparse
import pickle
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path")
    parser.add_argument("--prefix", default="")

    args = parser.parse_args()

    from pathlib import Path

    for path in tqdm(list(Path(args.input_path).glob(f"{args.prefix}*.pkl"))):
        with open(path, "rb") as handle:
            data = pickle.load(handle)

        if len(list(set(data.keys()).intersection(set(["x_train", "x_test"])))) > 0:
            print(data.keys())

            data.pop("x_train")
            data.pop("x_test")


        if len(list(set(data.keys()).intersection(set(["smiles_train", "smiles_test"])))) > 0:
            print(data.keys())

            data.pop("x_train")
            data.pop("x_test")
        with open(path, "wb") as handle:
            pickle.dump(data, handle)

        del data


if __name__ == "__main__":
    main()

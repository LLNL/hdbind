################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import time
import torch
import random
# import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

# from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
# import deepchem as dc
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm



def timeit_cpu_cuda(func, *args, **kwargs):

    cuda_starter, cuda_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()  # Record the start time
    cuda_starter.record()
    result = func(*args, **kwargs)  # Call the original function
    cuda_ender.record()
    torch.cuda.synchronize()
    end_time = time.perf_counter()  # Record the end time


    cpu_time = end_time - start_time
    cuda_time = cuda_starter.elapsed_time(cuda_ender) / 1000 # convert to milliseconds
    # print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to run.")
    return result, cpu_time, cuda_time




def seed_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_list_fn(data):
    return [x for x in data]
def get_random_hv(m, n):
    return torch.bernoulli(torch.tensor([[0.5] * m] * n)).float() * 2 - 1


class timing_part:
    def __init__(self, TAG, verbose=False):
        self.TAG = str(TAG)
        self.total_time = 0
        self.start_time = 0
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        exit_time = time.perf_counter()
        self.total_time = exit_time - self.start_time
        if self.verbose:
            tqdm.write(f"{self.TAG}\t{self.total_time}")


class TimerCudaCpu:
    def __init__(self, TAG, verbose=False):
        self.TAG = str(TAG)
        self.total_time_cpu = 0.0
        self.total_time_cuda = 0.0
        self.start_time = 0.0
        self.verbose = verbose
        self.cuda_starter = torch.cuda.Event(enable_timing=True)
        self.cuda_ender = torch.cuda.Event(enable_timing=True)


    def __enter__(self):
        self.start_time = time.perf_counter()
        self.starter.record()
        return self

    def __exit__(self, type, value, traceback):
        self.cuda_ender.record()
        torch.cuda.synchronize()
        exit_time = time.perf_counter()
        self.total_time_cpu = exit_time - self.start_time
        self.total_time_cuda = self.cuda_starter.elapsed_time(self.cuda_ender)
        if self.verbose:
            tqdm.write(f"{self.TAG}\tcpu_time: {self.total_time_cpu}, cuda_time: {self.total_time_cuda}")

def load_molformer_embeddings(embed_path):
    data = torch.load(embed_path)
    return data["embeds"], data["labels"]


def load_features(path: str, dataset: str):
    features, labels = None, None
    data = np.load(path)

    if dataset == "pdbbind":
        # map the experimental -logki/kd value to a discrete category
        features = data[:, :-1]
        labels = data[:, -1]
        labels = np.asarray(
            [convert_pdbbind_affinity_to_class_label(x) for x in labels]
        )

        binary_label_mask = labels != 2

        return features[binary_label_mask], labels[binary_label_mask]

    elif dataset == "dude":
        features = data[:, :-1]
        labels = data[:, -1]

        labels = 1 - labels
    else:
        raise NotImplementedError("specify a valid supported dataset type")

    # import ipdb
    # ipdb.set_trace()
    return features, labels


def bipolarize(hv):
    # hv = torch.where(hv <= 0, hv, -1).int()
    # hv = torch.where(hv > 0, hv, 1).int()

    # return hv
    return torch.where(hv > 0, torch.tensor(1.0, device=hv.device), 
                                    torch.tensor(-1.0, device=hv.device))

# rename this after switch existing calls from binarize to bipolarize
def binarize_(hv):
    
    # Convert positive values to 1, and non-positive values (0 or negative) to -1
    return torch.where(hv > 0, torch.tensor(1.0, device=hv.device), 
                                    torch.tensor(0.0, device=hv.device))




def tok_seq_to_hv(tokens: list, D: int, item_mem: dict):
    # hv = torch.zeros(D).int()

    hv = np.zeros(D).astype(int)

    # for each token in the sequence, retrieve the hv corresponding to it
    # then rotate the tensor elements by the position number the token
    # occurs at in the sequence. add to (zero-initialized hv representing the
    for idx, token in enumerate(tokens):
        token_hv = item_mem[token]
        hv = hv + np.roll(token_hv, idx).astype(int)

    hv = np.where(hv > 0, hv, -1).astype(int)
    hv = np.where(hv <= 0, hv, 1).astype(int)

    return hv


# def compute_splits(
    # split_path: Path,
    # random_state: int,
    # split_type: str,
    # df: pd.DataFrame,
    # smiles_col: str,
    # label_col: str,
# ):
    # reset the index of the dataframe
    # df = df.reset_index()

    # split_df = None
    # if not split_path.exists():
        # print(f"computing split file: {split_path}")
        # if split_type == "random":
            # train_idxs, test_idxs = train_test_split(
                # list(range(len(df))), random_state=random_state
            # )

        # elif split_type == "scaffold":
            # scaffoldsplitter = dc.splits.ScaffoldSplitter()
            # idxs = np.array(list(range(len(df))))

            #todo: if dataset
            # if label_col is None:
                # dataset = dc.data.DiskDataset.from_numpy(
                    # X=idxs, y=np.zeros(len(idxs), 1), ids=df[smiles_col].values
                # )
            # else:
                # dataset = dc.data.DiskDataset.from_numpy(
                    # X=idxs, y=df[label_col], ids=df[smiles_col].values
                # )

            # import ipdb
            # ipdb.set_trace()
            # train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            # train_idxs = train_data.X
            # test_idxs = test_data.X

        # import ipdb
        # ipdb.set_trace()
        # create the train/test column
        # train_df = df.loc[train_idxs]
        # test_df = df.loc[test_idxs]
        # train_df["split"] = ["train"] * len(train_df)
        # test_df["split"] = ["test"] * len(test_df)

        # split_df = pd.concat([train_df, test_df])
        # split_df.to_csv(split_path, index=True)

    # else:
        # print(f"split path: {split_path} exists. loading.")
        # split_df = pd.read_csv(split_path, index_col=0)

    # return split_df


def load_pdbbind_from_hdf(
    hdf_path,
    dataset_name,
    train_split_path,
    test_split_path,
    bind_thresh,
    no_bind_thresh,
):
    """
    input: parameters (dataset name, threshold values, train/test splits) and h5 file containing data and binding measurements
    output: numpy array with features
    """

    f = h5py.File(hdf_path, "r")

    # need an argument for train split, need an argument for test split

    train_df = pd.read_csv(train_split_path)
    test_df = pd.read_csv(test_split_path)

    train_ids = train_df["pdbid"].values.tolist()
    test_ids = test_df["pdbid"].values.tolist()

    train_list = []
    test_list = []

    for key in tqdm(list(f), total=len(list(f))):
        if key in train_ids:
            train_list.append(key)
        elif key in test_ids:
            test_list.append(key)
        else:
            print(f"key: {key} not contained in train or test split")
            continue

    train_data_list = []
    train_label_list = []
    for key in train_list:
        affinity = f[key].attrs["affinity"]

        if affinity > bind_thresh:
            train_label_list.append(1)
        elif affinity < no_bind_thresh:
            train_label_list.append(0)
        else:
            print(f"key: {key} has ambiguous label")
            continue

        train_data_list.append(np.asarray(f[key][dataset_name]))

    test_data_list = []
    test_label_list = []
    for key in test_list:
        affinity = f[key].attrs["affinity"]

        if affinity > bind_thresh:
            test_label_list.append(1)
        elif affinity < no_bind_thresh:
            test_label_list.append(0)
        else:
            print(f"key: {key} has ambiguous label")
            continue

        test_data_list.append(np.asarray(f[key][dataset_name]))

    return (
        np.asarray(train_data_list),
        np.asarray(train_label_list),
        np.asarray(test_data_list),
        np.asarray(test_label_list),
    )



def dump_dataset_to_disk(dataset:Dataset, output_path:Path):
    assert output_path is not None

    dataloader = DataLoader(dataset, num_workers=8, batch_size=128)

    if not output_path.exists():
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        
        hv_list = []
        label_list = []
        for batch in tqdm(dataloader, desc="writing hypervectors to disk.."):
            hv, label = batch

            hv_list.append(hv.cpu())
            label_list.append(label.cpu())
        
        hvs = torch.cat(hv_list)
        labels = torch.cat(label_list).reshape(-1,1)

        data = torch.cat([hvs, labels], dim=1).numpy()
        np.save(output_path, data)

    else:
        print(f"{output_path} exists! skipping.")





def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf-path", help="path to input hdf file containing pdbbind data"
    )
    parser.add_argument(
        "--dataset-name",
        nargs="+",
        help="sequence of dataset names to use from the hdf-path dataset",
    )
    parser.add_argument(
        "--train-split-path",
        help="path to list of pdbids corresponding to the training set",
    )
    parser.add_argument(
        "--test-split-path",
        help="path to list of pdbids corresponding to the testing set",
    )
    parser.add_argument(
        "--bind-thresh",
        type=float,
        help="threshold (lower) to use for determining binders from experimental measurement",
    )
    parser.add_argument(
        "--no-bind-thresh",
        type=float,
        help="threshold (upper) to use for determining non-binders from experimental measurement",
    )
    parser.add_argument("--run-HD-benchmark", action="store_true")
    args = parser.parse_args()

    for dataset_name in args.dataset_name:
        data = load_pdbbind_from_hdf(
            hdf_path=args.hdf_path,
            dataset_name=dataset_name,
            train_split_path=args.train_split_path,
            test_split_path=args.test_split_path,
            bind_thresh=args.bind_thresh,
            no_bind_thresh=args.no_bind_thresh,
        )

        x_train, y_train, x_test, y_test = data

        # import pdb
        # pdb.set_trace()
        if args.run_HD_benchmark:
            from hdpy.tfHD import train_test_loop

            train_test_loop(
                x_train.squeeze(),
                x_test.squeeze(),
                y_train,
                y_test,
                iterations=10,
                dimensions=10000,
                Q=10,
                K=2,
                batch_size=32,
                sim_metric="cos",
            )

    print(data)


def convert_pdbbind_affinity_to_class_label(x, pos_thresh=8, neg_thresh=6):
    if x < neg_thresh:
        return 0
    elif x > pos_thresh:
        return 1
    else:
        return 2


if __name__ == "__main__":
    main()

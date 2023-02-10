from cProfile import run
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

# import multiprocessing as mp
import torch.multiprocessing as mp

# torch.multiprocessing.set_sharing_strategy("file_system")
import deepchem as dc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from rdkit import Chem
from hdpy.ecfp_hd.encode_ecfp import ECFPEncoder
from hdpy.baseline_hd.classification_modules import RPEncoder
from hdpy.rff_hdc.encoder import RandomFourierEncoder
from hdpy.mole_hd.encode_smiles import SMILESHDEncoder, tokenize_smiles
from hdpy.ecfp_hd.encode_ecfp import compute_fingerprint_from_smiles
from hdpy.metrics import validate

import random

random.seed(0)
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--smiles', nargs='*', default=['CC[N+](C)(C)Cc1ccccc1Br'])
parser.add_argument("--ngram-order", type=int, default=0, help="specify the ngram order, 1-unigram, 2-bigram, so on. 0 is default to trigger an error in case ngram is specified as the tokenizer, we don't use this arg for atomwise or bpe")
parser.add_argument("--tokenizer", choices=["atomwise", "ngram", "bpe"])
parser.add_argument("--D", type=int, help="size of encoding dimension", default=10000)
parser.add_argument(
    "--dataset",
    choices=[
        "bbbp",
        "sider",
        "clintox",
        "dude",
        "lit-pcba",
        "lit-pcba-ave",
        "dockstring",
    ],
    required=True,
)
parser.add_argument("--split-type", choices=["random", "scaffold"], required=True)
parser.add_argument(
    "--input-feat-size", type=int, help="size of input feature dim. ", default=1024
)
parser.add_argument("--model", choices=["smiles-pe", "ecfp", "rp", "rf", "mlp"])
parser.add_argument(
    "--n-trials", type=int, default=1, help="number of trials to perform"
)
parser.add_argument("--random-state", type=int, default=0)
parser.add_argument("--hd-retrain-epochs", type=int, default=1)
args = parser.parse_args()


def train(model, hv_train, y_train, epochs=10):

    # import pdb 
    # pdb.set_trace()
    model.build_am(hv_train, y_train)

    for _ in range(epochs):

        model.retrain(hv_train, y_train)


def test(model, hv_test, y_test):

    pred_list = model.predict(hv_test)

    eta_list = model.compute_confidence(hv_test)

    return {"y_pred": pred_list, "y_true": y_test, "eta": eta_list}


def functional_encode(datapoint):
    hv = torch.zeros(args.D)

    for pos, value in enumerate(datapoint):

        if isinstance(value, torch.Tensor):

            hv = (
                hv
                + hd_model.item_mem["pos"][pos]
                * hd_model.item_mem["value"][value.data.int().item()]
            )

        else:

            hv = hv + hd_model.item_mem["pos"][pos] * hd_model.item_mem["value"][value]

        # bind both item memory elements? or should I use a single 2 by n_bit matrix of values randomly chosen to associate with all possibilities?
        # hv = hv + (pos_hv * value_hv)

    # binarize
    hv = torch.where(hv > 0, hv, -1)
    hv = torch.where(hv <= 0, hv, 1)

    return hv


def collate_ecfp(batch):

    return torch.stack([functional_encode(x) for x in batch])


# def run_hd_trial(smiles, labels, train_idxs, test_idxs):
def run_hd_trial(x_train, y_train, x_test, y_test, smiles_train=None, smiles_test=None):


    if args.dataset in ["dude", "lit-pcba", "lit-pcba-ave", "dockstring"]:
        root_data_p = Path(f".hd_cache/{args.model}/{args.dataset}/{args.split_type}/{target_name}")

    elif args.dataset in ["sider", "clintox"]:
        root_data_p = Path(f".hd_cache/{args.model}/{args.dataset}/{args.split_type}/{task_idx}")

    else:
        root_data_p = Path(f".hd_cache/{args.model}/{args.dataset}/{args.split_type}")

    train_hv_p = root_data_p / Path("train_dataset_hv.pth")
    test_hv_p = root_data_p / Path("test_dataset_hv.pth")
    im_p = root_data_p / Path("item_mem.pth")

    train_encode_time = 0
    test_encode_time = 0

 
    if args.model == "smiles-pe":

        # smiles-pe is a special case because it tokenizes the string based on use specified parameters
        train_hv_p = root_data_p / Path(
            f"{args.tokenizer}-{args.ngram_order}-train_dataset_hv.pth"
        )
        test_hv_p = root_data_p / Path(
            f"{args.tokenizer}-{args.ngram_order}-test_dataset_hv.pth"
        )
        im_p = root_data_p / Path(f"{args.tokenizer}-{args.ngram_order}-item_mem.pth")

        train_encode_time = 0
        if not im_p.exists():

            train_toks = tokenize_smiles(
                smiles_train, tokenizer=args.tokenizer, ngram_order=args.ngram_order
            )
            test_toks = tokenize_smiles(
                smiles_test, tokenizer=args.tokenizer, ngram_order=args.ngram_order
            )

            im_p.parent.mkdir(parents=True, exist_ok=True)

            toks = train_toks + test_toks

            hd_model.build_item_memory(toks)
            train_encode_start = time.time()
            train_dataset_hvs = hd_model.encode_dataset(train_toks)
            train_encode_time = time.time() - train_encode_start

            test_encode_start = time.time()
            test_dataset_hvs = hd_model.encode_dataset(test_toks)
            test_encode_time = time.time() - test_encode_start

            train_dataset_hvs = torch.vstack(train_dataset_hvs).int()
            test_dataset_hvs = torch.vstack(test_dataset_hvs).int()

            torch.save(train_dataset_hvs, train_hv_p)
            torch.save(test_dataset_hvs, test_hv_p)
            torch.save(hd_model.item_mem, im_p)
        else:
            item_mem = torch.load(im_p)
            hd_model.item_mem = item_mem

            train_dataset_hvs = torch.load(train_hv_p)
            test_dataset_hvs = torch.load(test_hv_p)

    elif args.model == "ecfp":

        # build the item memory, checks to see if it exists first before doing so
        if not im_p.exists():
            im_p.parent.mkdir(parents=True, exist_ok=True)
            hd_model.build_item_memory(n_bits=x_train.shape[1])
            torch.save(hd_model.item_mem, im_p)
        else:
            item_mem = torch.load(im_p)
            hd_model.item_mem = item_mem

        # encoding block
        train_encode_time = 0
        train_dataset_hvs = None

        # maybe should instead use a flag (or use at a level above) to allow the user to specify cases where they might want to measure the latency of this step
        if not train_hv_p.exists():

            # ENCODE TRAINING
            train_hv_p.parent.mkdir(parents=True, exist_ok=True)
            train_encode_start = (
                time.time()
            )  # im putting this inside of the context manager to avoid overhead potentially
            train_dataloader = DataLoader(
                # x_train, num_workers=70, batch_size=1000, collate_fn=collate_ecfp
                x_train, num_workers=int(mp.cpu_count()-1/2), batch_size=1000, collate_fn=collate_ecfp
            )
            train_dataset_hvs = []
            for train_batch_hvs in tqdm(train_dataloader, total=len(train_dataloader)):
                train_dataset_hvs.append(train_batch_hvs)
            train_encode_time = time.time() - train_encode_start
            train_dataset_hvs = torch.cat(train_dataset_hvs, dim=0)
            torch.save(train_dataset_hvs, train_hv_p)

        else:
            train_dataset_hvs = torch.load(train_hv_p)

        if not test_hv_p.exists():

            # ENCODE TESTING
            test_hv_p.parent.mkdir(parents=True, exist_ok=True)
            # with mp.Pool(mp.cpu_count() -1) as p:
            test_encode_start = (
                time.time()
            )  # im putting this inside of the context manager to avoid overhead potentially
            test_dataloader = DataLoader(
                # x_test, num_workers=70, batch_size=1000, collate_fn=collate_ecfp
                x_test, num_workers=int(mp.cpu_count()-1/2), batch_size=1000, collate_fn=collate_ecfp
            )
            test_dataset_hvs = []
            for test_batch_hvs in tqdm(test_dataloader, total=len(test_dataloader)):
                # batch_hvs = list(tqdm(p.imap(functional_encode, batch), total=len(batch), desc=f"encoding ECFPs with {mp.cpu_count() -1} workers.."))
                test_dataset_hvs.append(test_batch_hvs)
            test_encode_time = time.time() - test_encode_start
            test_dataset_hvs = torch.cat(test_dataset_hvs, dim=0)
            torch.save(test_dataset_hvs, test_hv_p)

        else:
            test_dataset_hvs = torch.load(test_hv_p)

    elif args.model == "rp":
        hd_model.cuda()
        # todo: cache the actual model projection matrix too

        # encode training data
        if not train_hv_p.exists():

            train_hv_p.parent.mkdir(parents=True, exist_ok=True)
            # TODO: the current code is capturing alot of other overhead besides the encoding, fix that
            train_encode_start = time.time()
            train_dataloader = DataLoader(x_train, num_workers=0, batch_size=1000)
            train_dataset_hvs = []
            for batch_hvs in tqdm(train_dataloader, total=len(train_dataloader)):
                train_dataset_hvs.append(hd_model.encode(batch_hvs.cuda()).cpu())

            train_dataset_hvs = torch.cat(train_dataset_hvs, dim=0)
            train_encode_time = time.time() - train_encode_start

            torch.save(train_dataset_hvs, train_hv_p)
            print(f"encode train: {train_encode_time}")
        else:
            train_dataset_hvs = torch.load(train_hv_p)

        # encode testing data
        if not test_hv_p.exists():

            test_hv_p.parent.mkdir(parents=True, exist_ok=True)
            # TODO: the current code is capturing alot of other overhead besides the encoding, fix that
            test_encode_start = time.time()
            test_dataloader = DataLoader(x_test, num_workers=0, batch_size=1000)
            test_dataset_hvs = []
            for batch_hvs in tqdm(test_dataloader, total=len(test_dataloader)):
                test_dataset_hvs.append(hd_model.encode(batch_hvs.cuda()).cpu())

            test_dataset_hvs = torch.cat(test_dataset_hvs, dim=0)
            test_encode_time = time.time() - test_encode_start

            torch.save(test_dataset_hvs, test_hv_p)
            print(f"encode test: {test_encode_time}")

        else:
            test_dataset_hvs = torch.load(test_hv_p)

    hd_model.cuda()

    train_dataset_labels = torch.from_numpy(y_train)
    test_dataset_labels = torch.from_numpy(y_test)


    train_start = time.time()
    train(
        hd_model, train_dataset_hvs, train_dataset_labels, epochs=args.hd_retrain_epochs
    )
    train_time = time.time() - train_start
    # import ipdb
    # ipdb.set_trace()


    test_start = time.time()
    result_dict = test(hd_model, test_dataset_hvs, test_dataset_labels)
    test_time = time.time() - test_start

    result_dict["y_pred"] = result_dict["y_pred"].cpu().numpy()
    result_dict["eta"] = result_dict["eta"].cpu().numpy().reshape(-1, 1)
    result_dict["y_true"] = result_dict["y_true"].cpu().numpy()
    result_dict["train_time"] = train_time
    result_dict["test_time"] = test_time
    result_dict["train_encode_time"] = train_encode_time
    result_dict["test_encode_time"] = test_encode_time
    result_dict["encode_time"] = train_encode_time + test_encode_time
    result_dict["train_size"] = train_dataset_hvs.shape[0]
    result_dict["test_size"] = test_dataset_hvs.shape[0]

    result_dict["class_report"] = classification_report(
        y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]
    )

    try:
        result_dict["roc-auc"] = roc_auc_score(
            y_score=result_dict["eta"], y_true=test_dataset_labels.cpu().numpy()
        )

    except ValueError as e:
        result_dict["roc-auc"] = None
        print(e)
    # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

    print(result_dict["class_report"])
    print(f"roc-auc {result_dict['roc-auc']}")

    validate(
        labels=test_dataset_labels.cpu().numpy(),
        pred_labels=result_dict["y_pred"],
        pred_scores=result_dict["eta"],
    )

    return result_dict


# def run_sklearn_trial(smiles, labels, train_idxs, test_idxs):
def run_sklearn_trial(x_train, y_train, x_test, y_test):

    model = None
    search = None


    scoring = "f1"

    if args.model == "rf":

        param_dist_dict = {"criterion": ["gini", "entropy"],
                    "max_depth": [x for x in np.linspace(2,np.log2(y_train.shape[0]), 10, dtype=int)],
                    # "min_samples_leaf": [x for x in np.linspace(5, int(y_train.shape[0]/10), 10, dtype=int)],
                    # "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True],
                    "oob_score": [True],
                    "max_samples": [x for x in np.linspace(10, y_train.shape[0], 10, dtype=int)],
                    "random_state": [args.random_state],
                    # "ccp_alpha": [x for x in np.linspace(0,1, 10, dtype=np.float32)],
                    }

        search = RandomizedSearchCV(
                            RandomForestClassifier(), 
                            param_dist_dict,
                            n_iter=10,
                            cv=5,
                            refit=False,
                            scoring=scoring,
                            verbose=True,
                            n_jobs=int(mp.cpu_count()-1))


    elif args.model == "mlp":

        param_dist_dict = {
            "early_stopping": [True],
            "validation_fraction": [0.1, 0.2],
            "n_iter_no_change": [2],
            "alpha": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "solver": ["adam"],
            "batch_size": np.linspace(8, 128, dtype=int),
            # "learning_rate": ['constant', 'invscaling', 'adaptive'],
            "activation": ['tanh', 'relu'],
            "random_state": [args.random_state], 
            "hidden_layer_sizes": [(512, 256, 128), (256, 128, 64), (128, 64, 32)],
            "verbose": [True]
        }

        # tl;dr these are the "larger" datasets so set max_iter to a smaller value in that case
        if args.dataset in ["lit-pcba", "lit-pcba-ave", "dockstring"]:
            param_dist_dict["max_iter"] = [10]

        else:
            param_dist_dict["max_iter"] = [100]

        search = RandomizedSearchCV(
                            MLPClassifier(), 
                            param_dist_dict,
                            n_iter=10,
                            cv=5,
                            refit=False,
                            scoring=scoring,
                            verbose=True,
                            n_jobs=int(mp.cpu_count()-1))


    else:
        raise NotImplementedError("please supply valid model type")



    # run the hyperparameter search (without refitting to full training set)

    search.fit(x_train, y_train)

    # collect the best parameters and train on full training set, we capture timings wrt to the optimal configuration
    if args.model == "rf":
        model = RandomForestClassifier(**search.best_params_)

    elif args.model == "mlp":
        model = MLPClassifier(**search.best_params_)


    train_start = time.time()
    model.fit(x_train, y_train.ravel())
    train_time = time.time() - train_start

    test_start = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - test_start

    result_dict = {}
    result_dict["y_pred"] = y_pred
    result_dict["eta"] = model.predict_proba(x_test).reshape(-1, 2)
    result_dict["y_true"] = y_test
    result_dict["train_time"] = train_time
    result_dict["test_time"] = test_time
    result_dict["train_encode_time"] = 0
    result_dict["test_encode_time"] = 0
    result_dict["encode_time"] = 0
    result_dict["train_size"] = x_train.shape[0]
    result_dict["test_size"] = x_test.shape[0]

    result_dict["class_report"] = classification_report(
        y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]
    )

    try:
        result_dict["roc-auc"] = roc_auc_score(
            y_score=result_dict["eta"][:, 1], y_true=y_test
        )

    except ValueError as e:
        result_dict["roc-auc"] = None
        print(e)
    # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

    print(result_dict["class_report"])
    print(f"roc-auc {result_dict['roc-auc']}")

    # enrichment metrics

    validate(
        labels=y_test,
        pred_labels=result_dict["y_pred"],
        pred_scores=result_dict["eta"][:, 1],
    )

    return result_dict


# def main(smiles, labels, train_idxs, test_idxs):
def main(x_train, y_train, x_test, y_test, smiles_train=None, smiles_test=None):
    trial_dict = {}
    for trial in range(args.n_trials):
        if args.model in ["smiles-pe", "ecfp", "rp"]:
            # result_dict = run_hd_trial(smiles=smiles, labels=labels, train_idxs=train_idxs, test_idxs=test_idxs)
            result_dict = run_hd_trial(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                smiles_train=smiles_train,
                smiles_test=smiles_test,
            )
        else:
            # result_dict = run_sklearn_trial(smiles=smiles, labels=labels, train_idxs=train_idxs, test_idxs=test_idxs)
            result_dict = run_sklearn_trial(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
            )
        trial_dict[trial] = result_dict

    return trial_dict


if __name__ == "__main__":

    """
    each problem (bbbp, sider, clintox) have n binary tasks..we'll form a separate AM for each
    """

    if args.model == "smiles-pe":

        hd_model = SMILESHDEncoder(D=args.D)

    elif args.model == "ecfp":

        hd_model = ECFPEncoder(D=args.D)

    elif args.model == "rp":

        assert args.input_feat_size is not None
        assert args.D is not None
        hd_model = RPEncoder(input_size=args.input_feat_size, D=args.D, num_classes=2)

    output_result_dir = Path("results")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True)

    print(args)

    if args.dataset == "bbbp":

        df = pd.read_csv("BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv")

        # probably want to use MOE_smiles
        smiles = df["rdkitSmiles"].values.tolist()
        labels = df["p_np"].values.reshape(-1, 1)

        fps = np.vstack([compute_fingerprint_from_smiles(x) for x in tqdm(smiles)])
        n_tasks = 1

        train_idxs, test_idxs = None, None 

        if args.split_type == "random":
            train_idxs, test_idxs = train_test_split(
                list(range(len(df))), random_state=args.random_state
            )

        elif args.split_type == "scaffold":

            scaffoldsplitter = dc.splits.ScaffoldSplitter()
            idxs = np.array(list(range(len(df))))

            dataset = dc.data.DiskDataset.from_numpy(
                X=idxs, w=np.zeros(len(df)), ids=df["rdkitSmiles"]
            )
            train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            train_idxs = train_data.X
            test_idxs = test_data.X

        x_train = fps[train_idxs, :]
        x_test = fps[test_idxs, :]


        y_train = labels[train_idxs]
        y_test = labels[test_idxs]

        smiles_train = df["rdkitSmiles"].iloc[train_idxs].values.tolist()
        smiles_test = df["rdkitSmiles"].iloc[test_idxs].values.tolist()


        output_file = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

        if output_file.exists():
            pass
        else:
            result_dict = main(x_train=x_train, y_train=y_train, 
                                x_test=x_test, y_test=y_test,
                                smiles_train=smiles_train, smiles_test=smiles_test)

            with open(output_file, "wb") as handle:
                pickle.dump(result_dict, handle)

    elif args.dataset == "sider":

        df = pd.read_csv("/g/g13/jones289/workspace/hd-cuda-master/datasets/sider.csv")

        label_cols = [x for x in df.columns.values if "smiles" not in x]
        smiles = df["smiles"].values.tolist()



        # make the fingerprints


        fps = np.vstack([compute_fingerprint_from_smiles(x) for x in tqdm(smiles)])



        labels = df[[x for x in df.columns.values if "smiles" not in x]].values
        n_tasks = len(label_cols)

        train_idxs = None
        test_idxs = None

        if args.split_type == "random":

            train_idxs, test_idxs = train_test_split(
                list(range(len(df))), random_state=args.random_state
            )
        elif args.split_type == "scaffold":

            scaffoldsplitter = dc.splits.ScaffoldSplitter()
            idxs = np.array(list(range(len(df))))

            dataset = dc.data.DiskDataset.from_numpy(
                X=idxs, w=np.zeros(len(df)), ids=df["smiles"]
            )
            train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            train_idxs = train_data.X
            test_idxs = test_data.X


        smiles_train = df["smiles"].iloc[train_idxs].values.tolist()
        smiles_test = df["smiles"].iloc[test_idxs].values.tolist()

        for task_idx in range(n_tasks):

            output_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.task_{task_idx}.{args.split_type}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

            if output_file.exists():
                pass
            else:
                x_train, x_test = fps[train_idxs, :], fps[test_idxs, :]
                y_train, y_test = labels[train_idxs, task_idx], labels[test_idxs, task_idx] 

                result_dict = main(x_train=x_train, x_test=x_test, 
                                    y_train=y_train, y_test=y_test, smiles_train=smiles_train,
                                    smiles_test=smiles_test)


            with open(
                output_file,
                "wb",
            ) as handle:
                pickle.dump(result_dict, handle)

    elif args.dataset == "clintox":


        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/clintox.csv"
        )


        # some of the smiles (6) for clintox dataset are not able to be converted to fingerprints, so skip them for all cases

        fps = [compute_fingerprint_from_smiles(x) for x in tqdm(df["smiles"].values.tolist())]
        valid_idxs = np.array([idx for idx, x in enumerate(fps) if x is not None])

        df = df.iloc[valid_idxs]

        label_cols = [x for x in df.columns.values if "smiles" not in x]

        smiles = df["smiles"].values.tolist()
        labels = df[label_cols].values
        n_tasks = len(label_cols)

        fps = [compute_fingerprint_from_smiles(x) for x in tqdm(smiles)]

        # some of the smiles (6) for clintox dataset are not able to be converted to fingerprints, so skip them for all cases
        # valid_idxs = np.array([idx for idx, x in enumerate(fps) if x is not None])


        fps = np.vstack(fps)

        train_idxs = None
        test_idxs = None

        if args.split_type == "random":

            train_idxs, test_idxs = train_test_split(
                list(range(len(df))), random_state=args.random_state
            )
        elif args.split_type == "scaffold":

            scaffoldsplitter = dc.splits.ScaffoldSplitter()

            # had some issues with a few of these molecules so doing a filter for rdkit validity
            valid_list = [
                (idx, x)
                for idx, x in enumerate(df["smiles"].values.tolist())
                if Chem.MolFromSmiles(x) is not None
            ]

            print(
                f"dropped {len(df) - len(valid_list)} invalid smiles for scaffold splitting"
            )

            dataset = dc.data.DiskDataset.from_numpy(
                X=[x[0] for x in valid_list],
                w=np.zeros(len(valid_list)),
                ids=[x[1] for x in valid_list],
            )

            train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            train_idxs = train_data.X
            test_idxs = test_data.X


        x_train = fps[train_idxs, :]
        x_test = fps[test_idxs, :]
        y_train = labels[train_idxs]
        y_test = labels[test_idxs]
        smiles_train = df["smiles"].iloc[train_idxs].values.tolist()
        smiles_test = df["smiles"].iloc[test_idxs].values.tolist()

        for task_idx in range(n_tasks):
            
            output_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.task_{task_idx}.{args.split_type}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

            if output_file.exists():
                pass 
            else:
                result_dict = main(
                    x_train=x_train, x_test=x_test, y_train=y_train[:, task_idx], y_test=y_test[:, task_idx], smiles_train=smiles_train, smiles_test=smiles_test
                )
                with open( 
                    output_file, "wb"
                ) as handle:

                    pickle.dump(result_dict, handle)

    elif args.dataset == "dude":

        if args.split_type == "scaffold":
            raise NotImplementedError(f"DUD-E does not support {args.split_type}")

        dude_data_p = Path("/usr/workspace/atom/gbsa_modeling/dude_smiles/")
        for dude_smiles_path in dude_data_p.glob("*_gbsa_smiles.csv"):

            target_name = dude_smiles_path.name.split("_")[0]



            output_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{args.split_type}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

            print(dude_smiles_path)
            if output_file.exists():
                pass 
            else:

                smiles_df = pd.read_csv(dude_smiles_path)

                dude_split_path = None
                if args.split_type == "random":
                    dude_split_path = dude_smiles_path.with_name(
                        dude_smiles_path.stem
                        + "_with_base_rdkit_smiles_train_valid_test_random_random.csv"
                    )

                split_df = pd.read_csv(dude_split_path)

                df = pd.merge(smiles_df, split_df, left_on="id", right_on="cmpd_id")

                train_idxs, test_idxs = [], []

                for grp_name, grp_df in df.groupby("subset"):
                    if grp_name == "train":
                        train_idxs = grp_df.index.values.tolist()
                    elif grp_name == "test":
                        test_idxs = grp_df.index.values.tolist()

                smiles = df["smiles"].values.tolist()
                labels = df["decoy"].apply(lambda x: int(not x)).values.reshape(-1, 1)

                n_tasks = 1
                ecfp_path = dude_smiles_path.parent / Path(f"{target_name}.{args.split_type}_ecfp.npy")

                fps = None
                if ecfp_path.exists():
                    fps = np.load(ecfp_path)

                else:
                    
                    fps = [compute_fingerprint_from_smiles(x) for x in tqdm(df["smiles"].values.tolist())]
                    fps = np.vstack(fps)
                    np.save(ecfp_path, fps)

                x_train, x_test = fps[train_idxs, :], fps[test_idxs, :]
                y_train, y_test = labels[train_idxs], labels[test_idxs]

                smiles_train = df["smiles"].iloc[train_idxs].values.tolist()
                smiles_test = df["smiles"].iloc[test_idxs].values.tolist()

                result_dict = main(
                    x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, smiles_train=smiles_train, smiles_test=smiles_test
                )

                with open(
                    output_file, "wb"
                ) as handle:
                    pickle.dump(result_dict, handle)

    elif args.dataset == "lit-pcba":

        lit_pcba_data_p = Path(
            "/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/"
        )
        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name

            print(lit_pcba_path)

            output_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

            if output_file.exists():
                # don't recompute if it's already been calculated
                pass  

            else:

                actives_df = pd.read_csv(
                    list(lit_pcba_path.glob("actives.smi"))[0],
                    header=None,
                    delim_whitespace=True,
                )
                actives_df["label"] = [1] * len(actives_df)

                inactives_df = pd.read_csv(
                    list(lit_pcba_path.glob("inactives.smi"))[0],
                    header=None,
                    delim_whitespace=True,
                )
                inactives_df["label"] = [0] * len(inactives_df)

                df = pd.concat([actives_df, inactives_df]).reset_index(drop=True)

                train_idxs, test_idxs = train_test_split(
                    list(range(len(df))),
                    stratify=df["label"],
                    random_state=args.random_state,
                )


                smiles = df[0].values.tolist()
                labels = df["label"].values.reshape(-1, 1)

                n_tasks = 1

                ecfp_path = lit_pcba_path.parent / Path(f"{target_name}/{args.split_type}_ecfp.npy")

                fps = None
                if ecfp_path.exists():
                    fps = np.load(ecfp_path)

                else:
                    
                    fps = [compute_fingerprint_from_smiles(x) for x in tqdm(df[0].values.tolist())]
                    fps = np.vstack(fps)
                    np.save(ecfp_path, fps)

                x_train, x_test = fps[train_idxs, :], fps[test_idxs, :]

                y_train, y_test = df["label"].values[train_idxs], df["label"].values[test_idxs]

                smiles_train = df[0].iloc[train_idxs].values.tolist()
                smiles_test = df[0].iloc[test_idxs].values.tolist()


                result_dict = main(
                    x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, smiles_train=smiles_train, smiles_test=smiles_test
                )

                with open(
                    output_file, "wb"
                ) as handle:
                    pickle.dump(result_dict, handle)

    elif args.dataset == "lit-pcba-ave":

        lit_pcba_data_p = Path(
            "/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/"
        )
        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name

            train_path = lit_pcba_path / Path("ecfp_train.npy")

            test_path = lit_pcba_path / Path("ecfp_test.npy")

            train_data = np.load(train_path)
            test_data = np.load(test_path)

            x_train = train_data[:, :-1]
            y_train = train_data[:, -1]

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            smiles_train = pd.read_csv(
                lit_pcba_path / Path("smiles_train.csv"), header=None
            )[0].values.tolist()

            smiles_test = pd.read_csv(
                lit_pcba_path / Path("smiles_test.csv"), header=None
            )[0].values.tolist()

            print(target_name, x_train.shape, y_train.shape, x_test.shape, y_test.shape)

            n_tasks = 1

            result_dict = main(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                smiles_train=smiles_train,
                smiles_test=smiles_test,
            )

            with open(
                f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl", "wb"
            ) as handle:
                pickle.dump(result_dict, handle)

    elif args.dataset == "dockstring":

        dockstring_p = Path(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/dockstring"
        )

        # for split in ["scaffold", "random"]:

        for path in (dockstring_p / Path(args.split_type)).glob("*"):
            print(dockstring_p)
            train_p = path / Path("train.npy")
            test_p = path / Path("test.npy")
            train_smiles_p = path / Path("train_smiles.npy")
            test_smiles_p = path / Path("test_smiles.npy")

            train_data = np.load(train_p)
            test_data = np.load(test_p)

            train_smiles = np.load(train_smiles_p, allow_pickle=True).flatten().tolist()
            test_smiles = np.load(test_smiles_p, allow_pickle=True).flatten().tolist()


            x_train = train_data[:, :-1]
            x_test = test_data[:, :-1]

            y_train = train_data[:, -1]
            y_test = test_data[:, -1]

            print(path)

            target_name = path.name

            result_dict = main(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, smiles_train=train_smiles, smiles_test=test_smiles
            )

            with open(
                f"{output_result_dir}/{args.dataset.replace('-', '_')}.{args.split_type}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl",
                "wb",
            ) as handle:
                pickle.dump(result_dict, handle)

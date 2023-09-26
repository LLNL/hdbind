from cProfile import run
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import selfies as sf
constrain_dict = sf.get_semantic_constraints()
# import multiprocessing as mp
import torch.multiprocessing as mp
from hdpy.utils import MolFormerDataset, ECFPDataset
# torch.multiprocessing.set_sharing_strategy("file_system")
import deepchem as dc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from rdkit import Chem
from hdpy.ecfp.encode import ECFPEncoder
from hdpy.baseline_hd.classification_modules import RPEncoder
# from hdpy.rff_hdc.encoder import RandomFourierEncoder
from hdpy.mole_hd.encode import SMILESHDEncoder, tokenize_smiles
from hdpy.selfies.encode import SELFIESHDEncoder, tokenize_selfies_from_smiles
from hdpy.ecfp.encode import compute_fingerprint_from_smiles
from hdpy.metrics import validate
from utils import load_molformer_embeddings
from hdpy.utils import compute_splits
# from hdpy.selfies.encode import encode_smiles_as_selfie
# import hdpy.parser as parser

'''
# todo: should use a dataset object for this
def compute_features_from_smiles(smiles, params):

    # import ipdb
    # ipdb.set_trace()
    if params["feature_type"] == "ecfp":

        feat = compute_fingerprint_from_smiles(
            smiles=smiles, input_size=params["input_size"], radius=params["ecfp_radius"]
        )

        return feat

    elif params["feature_type"] == "molformer":

        # load from molformer embedding directory?
        # todo: I need to use consistent splits so will need to load x_train and x_test simulaneously
        embed_dir = f"/p/lustre2/jones289/molformer_embeddings/{args.dataset}_molformer_embeddings.pkl"

        x = load_molformer_embeddings(embed_dir)
        # import ipdb
        # ipdb.set_trace()
        return x[0]
'''

# seed the RNGs
import random


def seed_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# seed_rngs(args.random_state)


def train(model, hv_train, y_train, epochs=10):

    single_pass_train_start = time.time()
    model.build_am(hv_train, y_train)
    single_pass_train_time = time.time() - single_pass_train_start

    learning_curve_list = []

    retrain_start = time.time()
    for _ in range(epochs):

        mistake_ct = model.retrain(hv_train, y_train, return_mistake_count=True)
        learning_curve_list.append(mistake_ct)

    retrain_time = time.time() - retrain_start
    return learning_curve_list, single_pass_train_time, retrain_time


def test(model, hv_test, y_test):

    # import pdb
    # pdb.set_trace()
    test_start = time.time()
    pred_list = model.predict(hv_test)
    test_time = time.time() - test_start

    conf_test_start = time.time()
    eta_list = model.compute_confidence(hv_test)
    conf_test_time = time.time() - conf_test_start

    return {
        "y_pred": pred_list,
        "y_true": y_test,
        "eta": eta_list,
        "test_time": test_time,
        "conf_test_time": conf_test_time,
    }


def functional_encode(datapoint):
    hv = torch.zeros(config.D)

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
def run_hd_trial(
    # root_data_p,
    hd_model,
    x_train,
    y_train,
    x_test,
    y_test,
    smiles_train=None,
    smiles_test=None,
):

    # train_hv_p = root_data_p / Path("train_dataset_hv.pth")
    # test_hv_p = root_data_p / Path("test_dataset_hv.pth")
    # im_p = root_data_p / Path("item_mem.pth")
    # am_p = root_data_p / Path("assoc_mem.pth")

    train_encode_time = 0
    test_encode_time = 0

    if config.model == "smiles-pe":

        train_encode_time = 0

        train_toks = tokenize_smiles(
            smiles_train, tokenizer=config.tokenizer, ngram_order=config.ngram_order, num_workers=1,
        )
        test_toks = tokenize_smiles(
            smiles_test, tokenizer=config.tokenizer, ngram_order=config.ngram_order, num_workers=1,
        )

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

    elif config.model == "selfies":

        # smiles-pe is a special case because it tokenizes the string based on use specified parameters
        # train_hv_p = root_data_p / Path(
            # f"{config.tokenizer}-{config.ngram_order}-train_dataset_hv.pth"
        # )
        # test_hv_p = root_data_p / Path(
            # f"{config.tokenizer}-{config.ngram_order}-test_dataset_hv.pth"
        # )
        # im_p = root_data_p / Path(f"{config.tokenizer}-{config.ngram_order}-item_mem.pth")

        # charwise = False
        # if config.tokenizer == "selfies_charwise":
            # charwise = True

        # train_encode_time = 0
        # if not im_p.exists():

            # train_toks = []

            # for smiles in smiles_train:
                # train_toks.append(
                    # tokenize_selfies_from_smiles(smiles, charwise=charwise)
                # )

            # test_toks = []
            # for smiles in smiles_test:
                # test_toks.append(
                    # tokenize_selfies_from_smiles(smiles, charwise=charwise)
                # )

            # im_p.parent.mkdir(parents=True, exist_ok=True)

            # toks = train_toks + test_toks

            # hd_model.build_item_memory(toks)
            # train_encode_start = time.time()
            # train_dataset_hvs = hd_model.encode_dataset(train_toks)
            # train_encode_time = time.time() - train_encode_start

            # test_encode_start = time.time()
            # test_dataset_hvs = hd_model.encode_dataset(test_toks)
            # test_encode_time = time.time() - test_encode_start

            # train_dataset_hvs = torch.vstack(train_dataset_hvs).int()
            # test_dataset_hvs = torch.vstack(test_dataset_hvs).int()

            # torch.save(train_dataset_hvs, train_hv_p)
            # torch.save(test_dataset_hvs, test_hv_p)
            # torch.save(hd_model.item_mem, im_p)
        # else:
            # item_mem = torch.load(im_p)
            # hd_model.item_mem = item_mem

            # train_dataset_hvs = torch.load(train_hv_p)
            # test_dataset_hvs = torch.load(test_hv_p)
            pass 

    elif config.model == "ecfp":

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
            train_dataloader = DataLoader(
                # x_train, num_workers=70, batch_size=1000, collate_fn=collate_ecfp
                x_train,
                num_workers=int(mp.cpu_count() - 1 / 2),
                batch_size=1000,
                collate_fn=collate_ecfp,
            )
            train_dataset_hvs = []
            train_encode_start = (
                time.time()
            )  # im putting this inside of the context manager to avoid overhead potentially
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
            test_dataloader = DataLoader(
                # x_test, num_workers=70, batch_size=1000, collate_fn=collate_ecfp
                x_test,
                num_workers=int(mp.cpu_count() - 1 / 2),
                batch_size=1000,
                collate_fn=collate_ecfp,
            )
            test_dataset_hvs = []
            test_encode_start = (
                time.time()
            )  # im putting this inside of the context manager to avoid overhead potentially
            for test_batch_hvs in tqdm(test_dataloader, total=len(test_dataloader)):
                # batch_hvs = list(tqdm(p.imap(functional_encode, batch), total=len(batch), desc=f"encoding ECFPs with {mp.cpu_count() -1} workers.."))
                test_dataset_hvs.append(test_batch_hvs)
            test_encode_time = time.time() - test_encode_start
            test_dataset_hvs = torch.cat(test_dataset_hvs, dim=0)
            torch.save(test_dataset_hvs, test_hv_p)

        else:
            test_dataset_hvs = torch.load(test_hv_p)

    elif config.model == "rp":
        # hd_model.cuda()
        # todo: cache the actual model projection matrix too

        # encode training data
        # if not train_hv_p.exists():

            # train_hv_p.parent.mkdir(parents=True, exist_ok=True)
            # TODO: the current code is capturing alot of other overhead besides the encoding, fix that
        train_encode_start = time.time()
        train_dataloader = DataLoader(x_train, num_workers=0, batch_size=1000)
        train_dataset_hvs = []
        # import ipdb
        # ipdb.set_trace()
        hd_model.to("cuda")
        for batch_hvs in tqdm(train_dataloader, total=len(train_dataloader)):
            train_dataset_hvs.append(hd_model.encode(batch_hvs.cuda()).cpu())

        train_dataset_hvs = torch.cat(train_dataset_hvs, dim=0)
        train_encode_time = time.time() - train_encode_start

        # torch.save(train_dataset_hvs, train_hv_p)
        print(f"encode train: {train_encode_time}")
        # else:
            # train_dataset_hvs = torch.load(train_hv_p)

        # encode testing data
        # if not test_hv_p.exists():

            # test_hv_p.parent.mkdir(parents=True, exist_ok=True)
            # TODO: the current code is capturing alot of other overhead besides the encoding, fix that
        test_encode_start = time.time()
        test_dataloader = DataLoader(x_test, num_workers=0, batch_size=1000)
        test_dataset_hvs = []
        for batch_hvs in tqdm(test_dataloader, total=len(test_dataloader)):
            test_dataset_hvs.append(hd_model.encode(batch_hvs.cuda()).cpu())

        test_dataset_hvs = torch.cat(test_dataset_hvs, dim=0)
        test_encode_time = time.time() - test_encode_start

            # import pdb
            # pdb.set_trace()
            # torch.save(test_dataset_hvs, test_hv_p)
            # print(f"encode test: {test_encode_time}")

        # else:
            # test_dataset_hvs = torch.load(test_hv_p)

    # create label tensors and transfer all tensors to the GPU
    # import ipdb
    # ipdb.set_trace()
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).to(device).float()
    if isinstance(y_test, np.ndarray):
        y_test = torch.from_numpy(y_test).to(device).float()
    # train_dataset_labels = 
    # test_dataset_labels = torch.from_numpy(y_test).to(device).float()

    train_dataset_hvs = train_dataset_hvs.to(device).float()
    test_dataset_hvs = test_dataset_hvs.to(device).float()

    result_dict = {}
    for i in range(args.n_trials):

        # this should force each call of .fit to be different...I think..
        seed_rngs(args.random_state + i)

        result_dict[i] = {}

        # time train inside of the funciton
        learning_curve, single_pass_train_time, retrain_time = train(
            hd_model,
            train_dataset_hvs,
            y_train,
            epochs=config.hd_retrain_epochs,
        )

        result_dict[i]["hd_learning_curve"] = learning_curve

        # import pdb
        # pdb.set_trace()
        # time test inside of the funcion
        trial_dict = test(hd_model, test_dataset_hvs, y_test)

        result_dict[i]["am"] = {
            0: hd_model.am[0].cpu().numpy(),
            1: hd_model.am[1].cpu().numpy(),
        }  # store the associative memory so it can be loaded up later on
        result_dict[i]["model"] = hd_model.to("cpu")
        result_dict[i]["y_pred"] = trial_dict["y_pred"].cpu().numpy()
        result_dict[i]["eta"] = trial_dict["eta"].cpu().numpy().reshape(-1, 1)
        result_dict[i]["y_true"] = trial_dict["y_true"].cpu().numpy()
        result_dict[i]["single_pass_train_time"] = single_pass_train_time
        result_dict[i]["retrain_time"] = retrain_time
        result_dict[i]["train_time"] = single_pass_train_time + retrain_time
        result_dict[i]["test_time"] = trial_dict["test_time"]
        result_dict[i]["conf_test_time"] = trial_dict["conf_test_time"]
        result_dict[i]["train_encode_time"] = train_encode_time
        result_dict[i]["test_encode_time"] = test_encode_time
        result_dict[i]["encode_time"] = train_encode_time + test_encode_time
        result_dict[i]["train_size"] = train_dataset_hvs.shape[0]
        result_dict[i]["test_size"] = test_dataset_hvs.shape[0]


        # import ipdb 
        # ipdb.set_trace()
        result_dict[i]["class_report"] = classification_report(
            y_pred=result_dict[i]["y_pred"], y_true=result_dict[i]["y_true"]
        )

        try:
            result_dict[i]["roc-auc"] = roc_auc_score(
                y_score=result_dict[i]["eta"], y_true=y_test.cpu().numpy()
            )

        except ValueError as e:
            result_dict[i]["roc-auc"] = None
            print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        print(result_dict[i]["class_report"])
        print(f"roc-auc {result_dict[i]['roc-auc']}")

        validate(
            labels=y_test.cpu().numpy(),
            pred_labels=result_dict[i]["y_pred"],
            pred_scores=result_dict[i]["eta"],
        )

    return result_dict


# def run_sklearn_trial(smiles, labels, train_idxs, test_idxs):
def run_sklearn_trial(x_train, y_train, x_test, y_test):

    model = None
    search = None

    scoring = "f1"

    if config.model == "rf":

        param_dist_dict = {
            "criterion": ["gini", "entropy"],
            "n_estimators": [x for x in np.linspace(10, 100, 10, dtype=int)],
            "max_depth": [x for x in np.linspace(2, x_train.shape[1], 10, dtype=int)],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 2, 5, 10],
            "bootstrap": [True],
            "oob_score": [True],
            "max_samples": [
                x for x in np.linspace(10, y_train.shape[0], 10, dtype=int)
            ],
            "n_jobs": [-1],
        }

        search = RandomizedSearchCV(
            RandomForestClassifier(),
            param_dist_dict,
            n_iter=10,
            cv=5,
            refit=False,
            scoring=scoring,
            verbose=True,
            n_jobs=int(mp.cpu_count() - 1),
        )

    elif config.model == "mlp":

        param_dist_dict = {
            "early_stopping": [True],
            "validation_fraction": [0.1, 0.2],
            "n_iter_no_change": [2],
            "alpha": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "solver": ["adam"],
            "batch_size": np.linspace(8, 128, dtype=int),
            "activation": ["tanh", "relu"],
            "hidden_layer_sizes": [(512, 256, 128), (256, 128, 64), (128, 64, 32)],
            "verbose": [True],
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
            n_jobs=int(mp.cpu_count() - 1),
        )

    else:
        raise NotImplementedError("please supply valid model type")

    # run the hyperparameter search (without refitting to full training set)

    search.fit(x_train, y_train)

    result_dict = {}
    for i in range(config.n_trials):

        seed = args.random_state + i
        # this should force each call of .fit to be different...I think..
        seed_rngs(seed)
        # collect the best parameters and train on full training set, we capture timings wrt to the optimal configuration
        # construct after seeding the rng
        if config.model == "rf":
            model = RandomForestClassifier(random_state=seed, **search.best_params_)

        elif config.model == "mlp":
            model = MLPClassifier(random_state=seed, **search.best_params_)

        result_dict[i] = {}

        train_start = time.time()
        model.fit(x_train, y_train.ravel())
        train_time = time.time() - train_start

        test_start = time.time()
        y_pred = model.predict(x_test)
        test_time = time.time() - test_start

        result_dict[i]["model"] = model
        result_dict[i]["search"] = search
        result_dict[i]["y_pred"] = y_pred
        result_dict[i]["eta"] = model.predict_proba(x_test).reshape(-1, 2)
        result_dict[i]["y_true"] = y_test
        result_dict[i]["train_time"] = train_time
        result_dict[i]["test_time"] = test_time
        result_dict[i]["train_encode_time"] = 0
        result_dict[i]["test_encode_time"] = 0
        result_dict[i]["encode_time"] = 0
        result_dict[i]["train_size"] = x_train.shape[0]
        result_dict[i]["test_size"] = x_test.shape[0]

        result_dict[i]["class_report"] = classification_report(
            y_pred=result_dict[i]["y_pred"], y_true=result_dict[i]["y_true"]
        )

        try:
            result_dict[i]["roc-auc"] = roc_auc_score(
                y_score=result_dict[i]["eta"][:, 1], y_true=y_test
            )

        except ValueError as e:
            result_dict[i]["roc-auc"] = None
            print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        print(result_dict[i]["class_report"])
        print(f"roc-auc {result_dict[i]['roc-auc']}")

        # enrichment metrics
        validate(
            labels=y_test,
            pred_labels=result_dict[i]["y_pred"],
            pred_scores=result_dict[i]["eta"][:, 1],
        )

    return result_dict


def main(
    hd_model,
    x_train,
    y_train,
    x_test,
    y_test,
    smiles_train=None,
    smiles_test=None,
):

    if config.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        result_dict = run_hd_trial(
            hd_model=hd_model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            smiles_train=smiles_train,
            smiles_test=smiles_test,
        )
    else:
        result_dict = run_sklearn_trial(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )

    return result_dict


def driver():


    # todo: ngram order and tokenizer only apply to some models, don't need to have in the exp_name
    exp_name = f"{Path(args.config).stem}"
    if config.model == "smiles-pe":

        hd_model = SMILESHDEncoder(D=config.D)

    elif config.model == "selfies":
        hd_model = SELFIESHDEncoder(D=config.D)

    elif config.model == "ecfp":

        hd_model = ECFPEncoder(D=config.D)

    elif config.model == "rp":

        assert config.input_size is not None
        assert config.D is not None
        hd_model = RPEncoder(input_size=config.input_size, D=config.D, num_classes=2)

    else:
        # if using sklearn or pytorch non-hd model
        hd_model = None

    if config.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        # transfer the model to GPU memory
        hd_model = hd_model.to(device).float()

        print("model is on the gpu")

    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(config)

    if args.dataset == "bbbp":

        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv"
        )

        train_idxs, test_idxs = None, None
        
        split_path = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        split_df = compute_splits(split_path=split_path,
                                  random_state=args.random_state,
                                  split_type=args.split_type,
                                  df=df,
                                  smiles_col="rdkitSmiles")
        

        x_train, x_test, y_train, y_test = None, None, None, None        
        smiles_train, smiles_test = None, None
        if config.embedding == "molformer":
            dataset = MolFormerDataset(path=f"/p/lustre2/jones289/molformer_embeddings/bbbp_molformer_embeddings.pt",
                                    split_df=split_df)

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            smiles_train, smiles_test = dataset.smiles_train, dataset.smiles_test

        if config.embedding == "ecfp":
            dataset = ECFPDataset(path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv",
                                smiles_col="rdkitSmiles", label_col="p_np",
                                #   split_df=df,
                                smiles=df["rdkitSmiles"],
                                split_df=split_df,
                                split_type=args.split_type,
                                random_state=args.random_state, 
                                ecfp_length=config.input_size,
                                ecfp_radius=config.ecfp_radius,
                                labels=df["p_np"].values)
            smiles_train = dataset.smiles_train
            smiles_test = dataset.smiles_test
            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()


        output_file = Path(
            f"{output_result_dir}/{exp_name}.pt"
        )

        if output_file.exists():
            print(f"{output_file} exists. skipping.")
            pass
        else:
            result_dict = main(
                # root_data_p=root_data_p,
                hd_model=hd_model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                smiles_train=smiles_train,
                smiles_test=smiles_test,
            )

            result_dict["smiles_train"] = smiles_train
            result_dict["smiles_test"] = smiles_test
            result_dict["x_train"] = x_train
            result_dict["x_test"] = x_test
            result_dict["y_train"] = y_train
            result_dict["y_test"] = y_test

            result_dict["args"] = config

            torch.save(result_dict, output_file)



        
    elif args.dataset == "sider":

        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/sider.csv"
        )


        split_path = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )


        split_df = compute_splits(split_path=split_path, 
                                  random_state=args.random_state, 
                                  split_type=args.split_type,
                                  df=df,
                                  smiles_col="smiles")


        label_list = [x for x in split_df.columns if x not in ['split', 'smiles', 'index']]
        for task_idx, label in enumerate(label_list):

            output_file = Path(
            f"{output_result_dir}/{exp_name}-{task_idx}.pkl"
            )

            if config.embedding == "molformer":
                dataset = MolFormerDataset(path=f"/p/lustre2/jones289/molformer_embeddings/sider-{task_idx}_molformer_embeddings.pt",
                                        split_df=split_df)

                x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                smiles_train, smiles_test = dataset.smiles_train, dataset.smiles_test

            if config.embedding == "ecfp":
                dataset = ECFPDataset(path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/sider.csv",
                                    smiles_col="smiles", label_col=label,
                                    #   split_df=df,
                                    smiles=df["smiles"],
                                    split_df=split_df,
                                    split_type=args.split_type,
                                    random_state=args.random_state, 
                                    ecfp_length=config.input_size,
                                    ecfp_radius=config.ecfp_radius,
                                    labels=df[label].values)
                smiles_train = dataset.smiles_train
                smiles_test = dataset.smiles_test
                x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                
    
            if output_file.exists():
                print(f"output_file: {output_file} exists. skipping.")
                pass
                
            else:
                result_dict = main(
                    hd_model=hd_model,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    smiles_train=smiles_train,
                    smiles_test=smiles_test,
                )

                result_dict["smiles_train"] = smiles_train
                result_dict["smiles_test"] = smiles_test
                result_dict["x_train"] = x_train
                result_dict["x_test"] = x_test
                result_dict["y_train"] = y_train
                result_dict["y_test"] = y_test

                result_dict["args"] = config
                with open(
                    output_file,
                    "wb",
                ) as handle:
                    pickle.dump(result_dict, handle)


    elif args.dataset == "clintox":

        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/clintox.csv"
        )

        # some of the smiles (6) for clintox dataset are not able to be converted to fingerprints, so skip them for all cases

        split_path = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )


        split_df = compute_splits(split_path=split_path, 
                                  random_state=args.random_state, 
                                  split_type=args.split_type,
                                  df=df,
                                  smiles_col="smiles")



        if config.embedding == "molformer":
            dataset = MolFormerDataset(path="/p/lustre2/jones289/molformer_embeddings/clintox_molformer_embeddings.pt",
                                       split_df=split_df)

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()

        if config.embedding == "ecfp":
            dataset = ECFPDataset(path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/clintox.csv",
                                  smiles_col="smiles", label_col="FDA_APPROVED",
                                #   split_df=df,
                                  smiles=df["smiles"],
                                  split_df=split_df,
                                  split_type=args.split_type,
                                  random_state=args.random_state, 
                                  ecfp_length=config.input_size,
                                  ecfp_radius=config.ecfp_radius,
                                  labels=df["FDA_APPROVED"].values)

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()

        output_file = Path(
            f"{output_result_dir}/{exp_name}.pkl"
        )

        if output_file.exists():
            print(f"output_file: {output_file} exists. skipping.")
            pass
        else:
            result_dict = main(
                hd_model=hd_model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                smiles_train=dataset.smiles_train,
                smiles_test=dataset.smiles_test,
            )

            result_dict["smiles_train"] = dataset.smiles_train
            result_dict["smiles_test"] = dataset.smiles_test
            result_dict["x_train"] = x_train
            result_dict["x_test"] = x_test
            result_dict["y_train"] = y_train
            result_dict["y_test"] = y_test

            result_dict["args"] = config
            with open(output_file, "wb") as handle:

                pickle.dump(result_dict, handle)

    elif args.dataset == "dude":

        if args.split_type.lower() != "random":  # i.e. scaffold
            raise NotImplementedError(f"DUD-E does not support {args.split_type}")

        dude_data_p = Path(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/dude_smiles/"
        )
        dude_path_list = list(dude_data_p.glob("*_gbsa_smiles.csv"))
        random.shuffle(dude_path_list)
        for dude_smiles_path in dude_path_list:

            target_name = dude_smiles_path.name.split("_")[0]

            output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name}.{args.random_state}.pkl"
        )

            if output_file.exists():

                print(
                    f"output file: {output_file} for input file {dude_smiles_path} already exists. moving on to next target.."
                )
                pass
            else:

                print(f"processing {dude_smiles_path}...")

                smiles_train_df, smiles_test_df = None, None
                dude_train_path = dude_smiles_path.with_name(
                    f"{dude_smiles_path.stem}_random_train.csv"
                )
                dude_test_path = dude_smiles_path.with_name(
                    f"{dude_smiles_path.stem}_random_test.csv"
                )


                smiles_df = pd.read_csv(dude_smiles_path)

                dude_split_path = None
                if args.split_type == "random":
                    dude_split_path = dude_smiles_path.with_name(
                        dude_smiles_path.stem
                        + "_with_base_rdkit_smiles_train_valid_test_random_random.csv"
                    )

                split_df = pd.read_csv(dude_split_path)

                df = pd.merge(smiles_df, split_df, left_on="id", right_on="cmpd_id")
                df['split'] = df['subset']
                df['index'] = df.index
                print(dude_train_path)

                if config.embedding == "smiles-raw":
                    pass

                elif config.embedding == "ecfp":
                    dataset = ECFPDataset(path=dude_smiles_path,
                                smiles_col="smiles", label_col="decoy",
                                #   split_df=df,
                                smiles=df["smiles"],
                                split_df=df,
                                split_type=args.split_type,
                                random_state=args.random_state, 
                                ecfp_length=config.input_size,
                                ecfp_radius=config.ecfp_radius,
                                labels=df["decoy"].astype(int).values)

                    x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                    '''
                    for split, smiles_list in [
                        ("train", smiles_train),
                        ("test", smiles_test),
                    ]:
                        ecfp_path = dude_smiles_path.parent / Path(
                            f"{target_name}.{args.split_type}.{split}.ecfp.npy"
                        )
                        fps = None
                        if ecfp_path.exists():
                            print(f"{ecfp_path} exists. loading from disk.")
                            fps = np.load(ecfp_path, allow_pickle=True)
                            print(f"done.")
                        else:

                            print(f"computing {ecfp_path}.")
                            fps = [
                                compute_fingerprint_from_smiles(
                                    x,
                                    input_size=config.input_size,
                                    radius=config.ecfp_radius,
                                )
                                for x in tqdm(smiles_list)
                            ]
                            fps = np.vstack(fps)
                            print("cacheing result..")
                            np.save(ecfp_path, fps)
                            print("done.")

                        if split == "train":
                            x_train = fps
                        if split == "test":
                            x_test = fps
                    '''
                elif config.embedding == "molformer":

                    dataset = MolFormerDataset(path=f"/p/lustre2/jones289/molformer_embeddings/dude-{target_name}_molformer_embeddings.pt",
                                       split_df=df)
                    # for split in ["train", "test"]:
                        # molformer_embed_path = dude_data_p / Path(
                            # f"{target_name}_gbsa_smiles_random_{split}_molformer_embeddings.pkl"
                        # )

                        # embeds = load_molformer_embeddings(molformer_embed_path)
                        # if split == "train":
                            # x_train, y_train = embeds
                        # if split == "test":
                            # x_test, y_test = embeds
                # x_train, x_test = fps[train_idxs, :], fps[test_idxs, :]
                # y_train, y_test = labels[train_idxs], labels[test_idxs]

                # smiles_train = df["smiles"].iloc[train_idxs].values.tolist()
                # smiles_test = df["smiles"].iloc[test_idxs].values.tolist()

                """
                np.save(output_file.with_name(f"{args.dataset}_{args.split_type}_{target_name}_labels_train.npy"), y_train)
                np.save(output_file.with_name(f"{args.dataset}_{args.split_type}_{target_name}_labels_test.npy"), y_test)
                """
                # import ipdb
                # ipdb.set_trace()

                result_dict = main(
                    hd_model=hd_model,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    smiles_train=dataset.smiles_train,
                    smiles_test=dataset.smiles_test,
                )

                result_dict["smiles_train"] = dataset.smiles_train
                result_dict["smiles_test"] = dataset.smiles_test
                result_dict["x_train"] = x_train
                result_dict["x_test"] = x_test
                result_dict["y_train"] = y_train
                result_dict["y_test"] = y_test

                result_dict["args"] = config
                with open(output_file, "wb") as handle:
                    pickle.dump(result_dict, handle)

            print(f"done. output file: {output_file}")

    elif args.dataset == "lit-pcba":

        lit_pcba_data_p = Path(
            "/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/"
        )

        # using random.shuffle so when submitting multiple jobs, chances are good that script can skip over previous work that has been done, increasing parallelism
        lit_pcba_path_list = list(lit_pcba_data_p.glob("*"))
        random.shuffle(lit_pcba_path_list)
        for lit_pcba_path in lit_pcba_path_list:

            target_name = lit_pcba_path.name
            output_file = Path(
                f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{config.model}.{config.tokenizer}.{config.ngram_order}.pkl"
            )

            if output_file.exists():
                # don't recompute if it's already been calculated
                print(
                    f"output file: {output_file} for input file {lit_pcba_path} already exists. moving on to next target.."
                )
                pass

            else:
                print(f"processing {lit_pcba_path}...")

                # import pdb
                # pdb.set_trace()

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

                ecfp_path = lit_pcba_path.parent / Path(
                    f"{target_name}/{args.split_type}_ecfp.npy"
                )

                fps = None
                if ecfp_path.exists():
                    fps = np.load(ecfp_path)

                else:

                    fps = [
                        compute_fingerprint_from_smiles(x)
                        for x in tqdm(df[0].values.tolist())
                    ]
                    fps = np.vstack(fps)
                    np.save(ecfp_path, fps)

                x_train, x_test = fps[train_idxs, :], fps[test_idxs, :]

                y_train, y_test = (
                    df["label"].values[train_idxs],
                    df["label"].values[test_idxs],
                )

                smiles_train = df[0].iloc[train_idxs].values.tolist()
                smiles_test = df[0].iloc[test_idxs].values.tolist()

                np.save(
                    output_file.with_name(
                        f"{args.dataset}_{args.split_type}_{target_name}_labels_train.npy"
                    ),
                    y_train,
                )
                np.save(
                    output_file.with_name(
                        f"{args.dataset}_{args.split_type}_{target_name}_labels_test.npy"
                    ),
                    y_test,
                )

                if not config.dry_run:
                    result_dict = main(
                        hd_model=hd_model,
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        y_test=y_test,
                        smiles_train=smiles_train,
                        smiles_test=smiles_test,
                    )

                    result_dict["smiles_train"] = smiles_train
                    result_dict["smiles_test"] = smiles_test
                    result_dict["x_train"] = x_train
                    result_dict["x_test"] = x_test
                    result_dict["y_train"] = y_train
                    result_dict["y_test"] = y_test

                    result_dict["args"] = config
                    with open(output_file, "wb") as handle:
                        pickle.dump(result_dict, handle)

                    print(f"done. output file: {output_file}")


if __name__ == "__main__":

    import argparser
    # args contains things that are unique to a specific run
    args = argparser.parse_args()
    # config contains general information about the model/data processing 
    config = argparser.get_config(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    driver()

################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
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
from hdpy.data_utils import MolFormerDataset, ECFPDataset, SMILESDataset
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
from hdpy.model import RPEncoder
from hdpy.molehd.encode import tokenize_smiles
from hdpy.selfies.encode import SELFIESHDEncoder
from hdpy.metrics import validate
from hdpy.utils import compute_splits
# from hdpy.selfies.encode import encode_smiles_as_selfie
# import hdpy.parser as parser
from torch.utils.data import TensorDataset
# from hdpy.model import get_random_hv
from hdpy.model import TokenEncoder
SCRATCH_DIR="/p/lustre2/jones289/"
SCRATCH_DIR="/p/vast1/jones289/"
# seed the RNGs
import random


def seed_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_list_fn(data):
    return [x for x in data]

def train(model, train_dataloader):
    
    with torch.no_grad():
        model = model.to(device)
        model.am = model.am.to(device)

        single_pass_train_time, retrain_time = None, None

        # build the associative memory with single-pass training
        
        for batch in tqdm(train_dataloader, desc=f"building AM with single-pass training.."):

            if config.model == "molehd": 
                x = [x[0] for x in batch]

                y = torch.from_numpy(np.array([x[1] for x in batch])).int()
            else:
                x, y = batch
        

            if not isinstance(x, list):
                x = x.to(device)

            for class_idx in range(2): #binary classification

                class_mask = y.squeeze() == class_idx

                if isinstance(x, list):

                    class_mask =class_mask.reshape(-1,1)
                    class_hvs = [model.encode(z) for z,w in zip(x, class_mask) if w == True]
                    if len(class_hvs) > 0:
                        class_hvs = torch.cat(class_hvs)
                        model.am[class_idx] += class_hvs.sum(dim=0)

                        #todo: have option to binarize the am after each update? or after all updates? or maybe just in the HDC model can have a flag that uses the exact AM versus the binarized AM 
                else:
                    model.am[class_idx] += model.encode(x[class_mask, :]).reshape(-1, config.D).sum(dim=0)

        learning_curve = []
        for epoch in range(config.epochs):
            
            mistake_ct = 0
            #TODO: initialize the associative memory with single pass training instead of the random initialization?
            for batch in tqdm(train_dataloader, desc=f"training-epoch {epoch}"):
                x, y, hv = None, None, None

                if config.model == "molehd":
                    x = [x[0] for x in batch]
                    y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                    y = y.squeeze().to(device)
                    hv = torch.cat([model.encode(z) for z in x])

                else:
                    x, y = batch
                    x, y = x.to(device), y.squeeze().to(device)
                    hv = model.encode(x)
                

                y_ = model.forward(hv)

                update_mask = torch.abs(y-y_).bool()
                mistake_ct += sum(update_mask)



                if update_mask.shape[0] == 1 and update_mask == False:
                    continue
                elif update_mask.shape[0] == 1 and update_mask == True:
                    # import ipdb
                    # ipdb.set_trace()
                    model.am[int(update_mask)] += hv.reshape(-1)
                    model.am[int(~update_mask.bool())] -= hv.reshape(-1)
                else:
                    for mistake_hv, mistake_label in zip(hv[update_mask], y[update_mask]):
                        # print(mistake_hv.shape,mistake_label.shape)
                        model.am[int(mistake_label)] += mistake_hv
                        model.am[int(~mistake_label.bool())] -= mistake_hv

            learning_curve.append(mistake_ct.cpu().numpy())

        return model, learning_curve, single_pass_train_time, retrain_time

def test(model, test_dataloader):

    with torch.no_grad():
        model = model.to(device)
        test_time_list = []
        test_encode_time_list = []
        conf_time_list = []
        target_list = []
        pred_list = []
        conf_list = []

        for batch in tqdm(test_dataloader, desc="testing.."):


            x, y, y_, hv, test_encode_end, test_encode_end = None, None, None, None, None, None
            if config.model == "molehd":
                x = [x[0] for x in batch]

                y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                y = y.squeeze()
                test_encode_start = time.time()
                hv = torch.cat([model.encode(z) for z in x])
                test_encode_end = time.time()
            else:
                x, y = batch
                x, y = x.to(device), y.squeeze().to(device)
                test_encode_start = time.time()
                hv = model.encode(x)
                test_encode_end = time.time()

            test_forward_start = time.time()
            y_ = model.forward(hv)
            test_forward_end = time.time()

            target_list.append(y.cpu().reshape(-1,1))
            pred_list.append(y_.cpu().reshape(-1,1))

            test_time_list.append(test_forward_end - test_forward_start)
            test_encode_time_list.append(test_encode_end - test_encode_start)

            conf_test_start = time.time()
            conf = model.compute_confidence(hv)
            conf_test_end = time.time() - conf_test_start
            conf_list.append(conf.cpu())
            conf_time_list.append(conf_test_end - conf_test_start)

        return {
            "y_pred": torch.cat(pred_list),
            "y_true": torch.cat(target_list),
            "eta": torch.cat(conf_list),
            "test_time": np.sum(test_time_list),
            "conf_test_time": np.sum(conf_time_list),
        }


def run_hd_trial(
    model,
    train_dataset, 
    test_dataset,
    smiles_train=None,
    smiles_test=None,
):

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

        model.build_item_memory(toks)
        train_encode_start = time.time()
        train_dataset_hvs = model.encode_dataset(train_toks)
        train_encode_time = time.time() - train_encode_start

        test_encode_start = time.time()
        test_dataset_hvs = model.encode_dataset(test_toks)
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

    collate_fn = None
    if config.model == "molehd":
        collate_fn = collate_list_fn

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers,
                                  persistent_workers=True,
                                  shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size,
                                 num_workers=1,
                                 persistent_workers=False,
                                 shuffle=False, collate_fn=collate_fn)

    result_dict = {"trials": {}}
    
    for i in range(args.n_trials):
        
        trial_dict = {}

        # this should force each call of .fit to be different...I think..
        seed_rngs(args.random_state + i)

        model, learning_curve, single_pass_train_time, retrain_time = train(model=model, train_dataloader=train_dataloader)


        trial_dict["hd_learning_curve"] = learning_curve

        # time test inside of the funcion
        test_dict = test(model, test_dataloader)

        trial_dict["am"] = {
            0: model.am[0].cpu().numpy(),
            1: model.am[1].cpu().numpy(),
        }  # store the associative memory so it can be loaded up later on
        # result_dict[i]["model"] = model.to("cpu")


        # '''
        trial_dict["y_pred"] = test_dict["y_pred"].cpu().numpy()
        trial_dict["eta"] = test_dict["eta"].cpu().numpy().reshape(-1, 1)
        trial_dict["y_true"] = test_dict["y_true"].cpu().numpy()
        trial_dict["single_pass_train_time"] = single_pass_train_time
        # result_dict[i]["retrain_time"] = retrain_time
        # result_dict[i]["train_time"] = single_pass_train_time + retrain_time
        trial_dict["test_time"] = test_dict["test_time"]
        trial_dict["conf_test_time"] = test_dict["conf_test_time"]
        trial_dict["train_encode_time"] = test_encode_time
        trial_dict["test_encode_time"] = test_encode_time
        trial_dict["encode_time"] = train_encode_time + test_encode_time
        # result_dict[i]["train_size"] = train_dataset_hvs.shape[0]
        # result_dict[i]["test_size"] = test_dataset_hvs.shape[0]



        trial_dict["class_report"] = classification_report(
            y_pred=trial_dict["y_pred"], y_true=trial_dict["y_true"]
        )

        try:
            trial_dict["roc-auc"] = roc_auc_score(
                y_score=trial_dict["eta"], y_true=trial_dict["y_true"]
            )

        except ValueError as e:
            trial_dict["roc-auc"] = None
            print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        print(trial_dict["class_report"])
        print(f"roc-auc {trial_dict['roc-auc']}")

        validate(
            labels=trial_dict["y_true"],
            pred_labels=trial_dict["y_pred"],
            pred_scores=trial_dict["eta"],
        )
        result_dict["trials"][i] = trial_dict
    # '''
    return result_dict
    


# def run_sklearn_trial(smiles, labels, train_idxs, test_idxs):
# def run_sklearn_trial(x_train, y_train, x_test, y_test):

    # model = None
    # search = None

    # scoring = "f1"

    # if config.model == "rf":

        # param_dist_dict = {
            # "criterion": ["gini", "entropy"],
            # "n_estimators": [x for x in np.linspace(10, 100, 10, dtype=int)],
            # "max_depth": [x for x in np.linspace(2, x_train.shape[1], 10, dtype=int)],
            # "max_features": ["sqrt", "log2"],
            # "min_samples_leaf": [1, 2, 5, 10],
            # "bootstrap": [True],
            # "oob_score": [True],
            # "max_samples": [
                # x for x in np.linspace(10, y_train.shape[0], 10, dtype=int)
            # ],
            # "n_jobs": [-1],
        # }

        # search = RandomizedSearchCV(
            # RandomForestClassifier(),
            # param_dist_dict,
            # n_iter=10,
            # cv=5,
            # refit=False,
            # scoring=scoring,
            # verbose=True,
            # n_jobs=int(mp.cpu_count() - 1),
        # )

    # elif config.model == "mlp":

        # param_dist_dict = {
            # "early_stopping": [True],
            # "validation_fraction": [0.1, 0.2],
            # "n_iter_no_change": [2],
            # "alpha": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            # "solver": ["adam"],
            # "batch_size": np.linspace(8, 128, dtype=int),
            # "activation": ["tanh", "relu"],
            # "hidden_layer_sizes": [(512, 256, 128), (256, 128, 64), (128, 64, 32)],
            # "verbose": [True],
        # }

        # tl;dr these are the "larger" datasets so set max_iter to a smaller value in that case
        # if args.dataset in ["lit-pcba", "lit-pcba-ave", "dockstring"]:
            # param_dist_dict["max_iter"] = [10]

        # else:
            # param_dist_dict["max_iter"] = [100]

        # search = RandomizedSearchCV(
            # MLPClassifier(),
            # param_dist_dict,
            # n_iter=10,
            # cv=5,
            # refit=False,
            # scoring=scoring,
            # verbose=True,
            # n_jobs=int(mp.cpu_count() - 1),
        # )

    # else:
        # raise NotImplementedError("please supply valid model type")

    # run the hyperparameter search (without refitting to full training set)

    # search.fit(x_train, y_train)

    #todo: (need to finsih refactoring this)
    # result_dict = {"trials": []}
    # for i in range(config.n_trials):


        # trial_dict = {}
        # seed = args.random_state + i
        # this should force each call of .fit to be different...I think..
        # seed_rngs(seed)
        # collect the best parameters and train on full training set, we capture timings wrt to the optimal configuration
        # construct after seeding the rng
        # if config.model == "rf":
            # model = RandomForestClassifier(random_state=seed, **search.best_params_)

        # elif config.model == "mlp":
            # model = MLPClassifier(random_state=seed, **search.best_params_)

        # result_dict[i] = {}

        # train_start = time.time()
        # model.fit(x_train, y_train.ravel())
        # train_time = time.time() - train_start

        # test_start = time.time()
        # y_pred = model.predict(x_test)
        # test_time = time.time() - test_start

        # result_dict[i]["model"] = model
        # result_dict[i]["search"] = search
        # result_dict[i]["y_pred"] = y_pred
        # result_dict[i]["eta"] = model.predict_proba(x_test).reshape(-1, 2)
        # result_dict[i]["y_true"] = y_test
        # result_dict[i]["train_time"] = train_time
        # result_dict[i]["test_time"] = test_time
        # result_dict[i]["train_encode_time"] = 0
        # result_dict[i]["test_encode_time"] = 0
        # result_dict[i]["encode_time"] = 0
        # result_dict[i]["train_size"] = x_train.shape[0]
        # result_dict[i]["test_size"] = x_test.shape[0]

        # result_dict[i]["class_report"] = classification_report(
            # y_pred=result_dict[i]["y_pred"], y_true=result_dict[i]["y_true"]
        # )

        # try:
            # result_dict[i]["roc-auc"] = roc_auc_score(
                # y_score=result_dict[i]["eta"][:, 1], y_true=y_test
            # )

        # except ValueError as e:
            # result_dict[i]["roc-auc"] = None
            # print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        # print(result_dict[i]["class_report"])
        # print(f"roc-auc {result_dict[i]['roc-auc']}")

        # enrichment metrics
        # validate(
            # labels=y_test,
            # pred_labels=result_dict[i]["y_pred"],
            # pred_scores=result_dict[i]["eta"][:, 1],
        # )

    # return result_dict


def main(
    model,
    train_dataset,
    test_dataset,
):

    if config.model in ["molehd", "selfies", "ecfp", "rp"]:
        result_dict = run_hd_trial(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
    else:
        result_dict = run_sklearn_trial(
            x_train=train_dataset.x, 
            y_train=train_dataset.y,
            x_test=test_dataset.x,
            y_test=test_dataset.y
        )

    return result_dict


def driver():


    # todo: ngram order and tokenizer only apply to some models, don't need to have in the exp_name
    exp_name = f"{Path(args.config).stem}"
    if config.model == "molehd":
        model = TokenEncoder(D=config.D, num_classes=2)
        # will update item_mem after processing input data


    elif config.model == "selfies":
        model = SELFIESHDEncoder(D=config.D)

    elif config.model == "ecfp":

        model = ECFPEncoder(D=config.D)

    elif config.model == "rp":

        assert config.input_size is not None
        assert config.D is not None
        model = RPEncoder(input_size=config.input_size, D=config.D, num_classes=2)

    else:
        # if using sklearn or pytorch non-hd model
        model = None

    if config.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        # transfer the model to GPU memory
        model = model.to(device).float()

        print("model is on the gpu")

    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(config)

    if args.dataset == "bbbp":
        # import ipdb
        # ipdb.set_trace()
        smiles_col = "rdkitSmiles"
        label_col = "p_np"
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
            # import ipdb
            # ipdb.set_trace()
            dataset = MolFormerDataset(path=f"{SCRATCH_DIR}/molformer_embeddings/bbbp_molformer_embeddings.pt",
                                    split_df=split_df,
                                    smiles_col="rdkitSmiles")

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            smiles_train, smiles_test = dataset.smiles_train, dataset.smiles_test
            train_dataset = TensorDataset(torch.from_numpy(x_train).int(), torch.from_numpy(y_train).int())
            test_dataset = TensorDataset(torch.from_numpy(x_test).int(), torch.from_numpy(y_test).int())





        elif config.embedding == "ecfp":
            dataset = ECFPDataset(path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv",
                                smiles_col=smiles_col, label_col=label_col,
                                #   split_df=df,
                                smiles=df[smiles_col],
                                split_df=split_df,
                                split_type=args.split_type,
                                random_state=args.random_state, 
                                ecfp_length=config.input_size,
                                ecfp_radius=config.ecfp_radius,
                                labels=df["p_np"].values)
            smiles_train = dataset.smiles_train
            smiles_test = dataset.smiles_test
            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            train_dataset = TensorDataset(x_train.int(), y_train.int())
            test_dataset = TensorDataset(x_test.int(), y_test.int())



        elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:

            
            train_smiles = split_df[smiles_col][split_df[split_df["split"] == "train"]["index"]]
            test_smiles = split_df[smiles_col][split_df[split_df["split"] == "test"]["index"]]

            train_labels = split_df[label_col][split_df[split_df["split"] == "train"]["index"]]
            test_labels = split_df[label_col][split_df[split_df["split"] == "test"]["index"]]

            train_dataset = SMILESDataset(smiles=train_smiles, labels=train_labels, 
                                            D=config.D, tokenizer=config.embedding, 
                                            ngram_order=config.ngram_order, num_workers=16,
                                            device=device)
            # use the item_memory generated by the train_dataset as a seed for the test, then update both?
            test_dataset = SMILESDataset(smiles=test_smiles, labels=test_labels, 
                                            D=config.D, tokenizer=config.embedding, 
                                            ngram_order=config.ngram_order,
                                            item_mem=train_dataset.item_mem, num_workers=1,
                                            device=device)
        
            train_dataset.item_mem = test_dataset.item_mem
            model.item_mem = train_dataset.item_mem
        else:
            raise NotImplementedError

        # this is really dumb, need to fix

        # output_file = Path(
            # f"{output_result_dir}/{exp_name}.pt"
        # )
        output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{args.split_type}.{args.random_state}.pkl"
        )
        if output_file.exists():
            print(f"{output_file} exists. skipping.")
            pass
        else:
            result_dict = main(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset
            )

            # result_dict["smiles_train"] = smiles_train
            # result_dict["smiles_test"] = smiles_test
            # result_dict["x_train"] = x_train
            # result_dict["x_test"] = x_test
            # result_dict["y_train"] = y_train
            # result_dict["y_test"] = y_test

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
                dataset = MolFormerDataset(path=f"{SCRATCH_DIR}/molformer_embeddings/sider-{task_idx}_molformer_embeddings.pt",
                                        split_df=split_df,
                                        smiles_col="smiles")

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
                    model=model,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    smiles_train=smiles_train,
                    smiles_test=smiles_test,
                )

                # result_dict["smiles_train"] = smiles_train
                # result_dict["smiles_test"] = smiles_test
                # result_dict["x_train"] = x_train
                # result_dict["x_test"] = x_test
                # result_dict["y_train"] = y_train
                # result_dict["y_test"] = y_test

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
            dataset = MolFormerDataset(path="{SCRATCH_DIR}/molformer_embeddings/clintox_molformer_embeddings.pt",
                                       split_df=split_df,
                                       smiles_col="smiles")

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)

        elif config.embedding == "ecfp":
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
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
        elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
            
            train_smiles = split_df[smiles_col][split_df[split_df["split"] == "train"]["index"]]
            test_smiles = split_df[smiles_col][split_df[split_df["split"] == "test"]["index"]]

            train_labels = split_df[label_col][split_df[split_df["split"] == "train"]["index"]]
            test_labels = split_df[label_col][split_df[split_df["split"] == "test"]["index"]]

            train_dataset = SMILESDataset(smiles=train_smiles, labels=train_labels, 
                                            D=config.D, tokenizer=config.embedding, 
                                            ngram_order=config.ngram_order, num_workers=16,
                                            device=device)
            # use the item_memory generated by the train_dataset as a seed for the test, then update both?
            test_dataset = SMILESDataset(smiles=test_smiles, labels=test_labels, 
                                            D=config.D, tokenizer=config.embedding, 
                                            ngram_order=config.ngram_order,
                                            item_mem=train_dataset.item_mem, num_workers=1,
                                            device=device)
        
            train_dataset.item_mem = test_dataset.item_mem
            model.item_mem = train_dataset.item_mem
        else:
            raise NotImplementedError




        output_file = Path(
            f"{output_result_dir}/{exp_name}.pkl"
        )

        if output_file.exists():
            print(f"output_file: {output_file} exists. skipping.")
            pass
        else:
            result_dict = main(
                model=model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                smiles_train=dataset.smiles_train,
                smiles_test=dataset.smiles_test,
            )

            # result_dict["smiles_train"] = dataset.smiles_train
            # result_dict["smiles_test"] = dataset.smiles_test
            # result_dict["x_train"] = x_train
            # result_dict["x_test"] = x_test
            # result_dict["y_train"] = y_train
            # result_dict["y_test"] = y_test

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

                if config.embedding == "ecfp":
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
                    train_dataset = TensorDataset(x_train, y_train)
                    test_dataset = TensorDataset(x_test, y_test)

                elif config.embedding == "molformer":

                    train_dataset = MolFormerDataset(path=f"{SCRATCH_DIR}/molformer_embeddings/dude-{target_name}_random_train_molformer_embeddings.pt",
                                       split_df=df, smiles_col="smiles")
                    test_dataset = MolFormerDataset(path=f"{SCRATCH_DIR}/molformer_embeddings/dude-{target_name}_random_test_molformer_embeddings.pt",
                                       split_df=df, smiles_col="smiles")

                result_dict = main(
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset
                )

                # result_dict["smiles_train"] = train_dataset.smiles
                # result_dict["smiles_test"] = test_dataset.smiles
                # result_dict["x_train"] = train_dataset.x
                # result_dict["x_test"] = test_dataset.x
                # result_dict["y_train"] = train_dataset.y
                # result_dict["y_test"] = test_dataset.y

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

            train_dataset, test_dataset = None, None

            target_name = lit_pcba_path.name
            
            # if target_name != "ESR1_ago":
                # continue
            # '''
            output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name}.{args.random_state}.pkl"
        )

            if output_file.exists():
                # don't recompute if it's already been calculated
                print(
                    f"output file: {output_file} for input file {lit_pcba_path} already exists. moving on to next target.."
                )
                pass

            else:
                print(f"processing {lit_pcba_path}...")

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

                _, test_idxs = train_test_split(
                    list(range(len(df))),
                    stratify=df["label"],
                    random_state=args.random_state,
                )

                df["smiles"] = df[0]
                df["index"] = df.index
                df["split"] = ["train"] * len(df)
                df.loc[test_idxs, "split"] = "test"

                # import ipdb
                # ipdb.set_trace()
                if config.embedding == "ecfp":
                    dataset = ECFPDataset(path=lit_pcba_path/Path("full_smiles_with_labels.csv"),
                                smiles_col="smiles", label_col="label",
                                #   split_df=df,
                                smiles=df["smiles"],
                                split_df=df,
                                split_type=args.split_type,
                                random_state=args.random_state, 
                                ecfp_length=config.input_size,
                                ecfp_radius=config.ecfp_radius,
                                labels=df["label"].astype(int).values)

                elif config.embedding == "molformer":

                    dataset = MolFormerDataset(path=f"{SCRATCH_DIR}/molformer_embeddings/lit-pcba-{target_name}_molformer_embeddings.pt",
                            split_df=df,
                            smiles_col="smiles")

                elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:

                    
                    train_smiles = df["smiles"][df[df["split"] == "train"]["index"]]
                    test_smiles = df["smiles"][df[df["split"] == "test"]["index"]]

                    train_labels = df["label"][df[df["split"] == "train"]["index"]]
                    test_labels = df["label"][df[df["split"] == "test"]["index"]]

                    train_dataset = SMILESDataset(smiles=train_smiles, labels=train_labels, 
                                                  D=config.D, tokenizer=config.embedding, 
                                                  ngram_order=config.ngram_order, num_workers=32,
                                                  device=device)
                    # use the item_memory generated by the train_dataset as a seed for the test, then update both?
                    test_dataset = SMILESDataset(smiles=test_smiles, labels=test_labels, 
                                                 D=config.D, tokenizer=config.embedding, 
                                                 ngram_order=config.ngram_order,
                                                 item_mem=train_dataset.item_mem, num_workers=1,
                                                 device=device)
                
                    train_dataset.item_mem = test_dataset.item_mem
                    model.item_mem = train_dataset.item_mem
                else:
                    raise NotImplementedError

                # this only applies to the molformer and ecfp input cases

                # import ipdb 
                # ipdb.set_trace()
                if config.embedding in ["molformer", "ecfp"]:
                    x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                    if isinstance(x_train, np.ndarray):
                        x_train, x_test, y_train, y_test = torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(y_train), torch.from_numpy(y_test)
                    train_dataset = TensorDataset(x_train, y_train)
                    test_dataset = TensorDataset(x_test, y_test)
                

                if output_file.exists():
                    with open(output_file, "rb") as handle:
                        result_dict = pickle.load(handle)

                else:
                    # continue
                    result_dict = main(
                        model=model,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset
                    )


                # import ipdb
                # ipdb.set_trace()
                if config.model == "molehd":
                    result_dict["y_train"] = (train_dataset.labels.values)
                    result_dict["y_test"] = (test_dataset.labels.values)
                else:
                    # result_dict["x_train"] = (train_dataset.tensors[0]).numpy()
                    # result_dict["x_test"] = (test_dataset.tensors[0]).numpy()
                    result_dict["y_train"] = (train_dataset.tensors[1]).numpy()
                    result_dict["y_test"] = (test_dataset.tensors[1]).numpy()

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

    if config.device == "cpu":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    driver()
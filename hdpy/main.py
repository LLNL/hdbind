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
import torch
import ray
from tqdm import tqdm
import numpy as np
import pandas as pd
import selfies as sf
constrain_dict = sf.get_semantic_constraints()
from hdpy.data_utils import MolFormerDataset, ECFPDataset, SMILESDataset
import deepchem as dc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from pathlib import Path

from hdpy.ecfp.encode import ECFPEncoder
from hdpy.model import RPEncoder, MLPClassifier
from hdpy.molehd.encode import tokenize_smiles
from hdpy.selfies.encode import SELFIESHDEncoder
from hdpy.metrics import validate
from hdpy.utils import compute_splits, collate_list_fn, seed_rngs
from torch.utils.data import TensorDataset
from hdpy.model import TokenEncoder

# seed the RNGs
import random
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.air import Checkpoint, RunConfig
from ray import train, tune
# from deepchem import molnet


# SCRATCH_DIR = "/p/lustre2/jones289/"
SCRATCH_DIR = "/p/vast1/jones289/"





def train_hdc_no_encode(model, train_dataloader, device, num_epochs):
    with torch.no_grad():
        model = model.to(device)
        model.am = model.am.to(device)

        single_pass_train_time, retrain_time = None, None

        # build the associative memory with single-pass training

        for batch in tqdm(
            train_dataloader, desc=f"building AM with single-pass training.."
        ):
            x, y = batch
            x = x.to("cuda")

            for class_idx in range(2):  # binary classification
                class_mask = y.squeeze() == class_idx

                model.am[class_idx] += (
                    (x[class_mask, :]).reshape(-1, model.D).sum(dim=0)
                )

        learning_curve = []
        train_encode_time_list = []

        for epoch in range(num_epochs):
            mistake_ct = 0
            # TODO: initialize the associative memory with single pass training instead of the random initialization?

            epoch_encode_time_total = 0
            for batch in tqdm(train_dataloader, desc=f"training HDC epoch {epoch}"):

                hv, y = batch
                hv = hv.to("cuda")
                y = y.to("cuda")
                y_ = model.predict(hv)
                update_mask = torch.abs(y - y_).bool()
                mistake_ct += sum(update_mask)

                if update_mask.shape[0] == 1 and update_mask == False:
                    continue
                elif update_mask.shape[0] == 1 and update_mask == True:
                    # import ipdb
                    # ipdb.set_trace()
                    model.am[int(update_mask)] += hv.reshape(-1)
                    model.am[int(~update_mask.bool())] -= hv.reshape(-1)
                else:
                    for mistake_hv, mistake_label in zip(
                        hv[update_mask], y[update_mask]
                    ):
                        # print(mistake_hv.shape,mistake_label.shape)
                        model.am[int(mistake_label)] += mistake_hv
                        model.am[int(~mistake_label.bool())] -= mistake_hv

            learning_curve.append(mistake_ct.cpu().numpy())
            train_encode_time_list.append(epoch_encode_time_total)

        return (
            model,
            learning_curve,
            single_pass_train_time,
            retrain_time,
            np.sum(train_encode_time_list) / num_epochs,
        )


# todo: this needs to be in the model.py
def train_hdc(model, train_dataloader, device, num_epochs):
    with torch.no_grad():
        model = model.to(device)
        model.am = model.am.to(device)

        single_pass_train_time, retrain_time = None, None

        # build the associative memory with single-pass training

        for batch in tqdm(
            train_dataloader, desc=f"building AM with single-pass training.."
        ):
            if model.name == "molehd":
                x = [x[0] for x in batch]

                y = torch.from_numpy(np.array([x[1] for x in batch])).int()
            else:
                x, y = batch

            if not isinstance(x, list):
                x = x.to(device)

            for class_idx in range(2):  # binary classification
                class_mask = y.squeeze() == class_idx

                if isinstance(x, list):
                    class_mask = class_mask.reshape(-1, 1)
                    class_hvs = [
                        model.encode(z) for z, w in zip(x, class_mask) if w == True
                    ]
                    if len(class_hvs) > 0:
                        class_hvs = torch.cat(class_hvs)
                        model.am[class_idx] += class_hvs.sum(dim=0)

                        # todo: have option to binarize the am after each update? or after all updates? or maybe just in the HDC model can have a flag that uses the exact AM versus the binarized AM
                else:
                    model.am[class_idx] += (
                        model.encode(x[class_mask, :]).reshape(-1, model.D).sum(dim=0)
                    )

        learning_curve = []
        train_encode_time_list = []

        # import pdb
        # pdb.set_trace()
        for epoch in range(num_epochs):
            mistake_ct = 0
            # TODO: initialize the associative memory with single pass training instead of the random initialization?

            epoch_encode_time_total = 0
            for batch in tqdm(train_dataloader, desc=f"training HDC epoch {epoch}"):
                x, y, hv = None, None, None

                if model.name == "molehd":
                    x = [x[0] for x in batch]
                    y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                    y = y.squeeze().to(device)
                    encode_time_start = time.time()
                    hv = torch.cat([model.encode(z) for z in x])
                    encode_time_end = time.time()
                    epoch_encode_time_total += encode_time_end - encode_time_start

                else:
                    x, y = batch
                    x, y = x.to(device), y.squeeze().to(device)
                    encode_time_start = time.time()
                    hv = model.encode(x)
                    encode_time_end = time.time()
                    epoch_encode_time_total += encode_time_end - encode_time_start

                # y_ = model.forward(hv)
                y_ = model.predict(hv)
                update_mask = torch.abs(y - y_).bool()
                mistake_ct += sum(update_mask)

                if update_mask.shape[0] == 1 and update_mask == False:
                    continue
                elif update_mask.shape[0] == 1 and update_mask == True:
                    # import ipdb
                    # ipdb.set_trace()
                    model.am[int(update_mask)] += hv.reshape(-1)
                    model.am[int(~update_mask.bool())] -= hv.reshape(-1)
                else:
                    for mistake_hv, mistake_label in zip(
                        hv[update_mask], y[update_mask]
                    ):
                        # print(mistake_hv.shape,mistake_label.shape)
                        model.am[int(mistake_label)] += mistake_hv
                        model.am[int(~mistake_label.bool())] -= mistake_hv

            learning_curve.append(mistake_ct.cpu().numpy())
            train_encode_time_list.append(epoch_encode_time_total)

        return (
            model,
            learning_curve,
            single_pass_train_time,
            retrain_time,
            np.sum(train_encode_time_list) / num_epochs,
        )


# todo: this needs to be in the model.py
def test_hdc(model, test_dataloader, device):
    with torch.no_grad():
        model = model.to(device)
        test_time_list = []
        test_encode_time_list = []
        conf_time_list = []
        target_list = []
        pred_list = []
        conf_list = []

        # import pdb
        # pdb.set_trace()
        for batch in tqdm(test_dataloader, desc="testing.."):
            x, y, y_, hv, test_encode_end, test_encode_end = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
            if model.name == "molehd":
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
            # y_ = model.forward(hv)
            y_ = model.predict(hv)
            test_forward_end = time.time()

            target_list.append(y.cpu().reshape(-1, 1))
            pred_list.append(y_.cpu().reshape(-1, 1))

            test_time_list.append(test_forward_end - test_forward_start)
            test_encode_time_list.append(test_encode_end - test_encode_start)

            conf_test_start = time.time()
            conf = model.compute_confidence(hv)
            conf_test_end = time.time()
            conf_list.append(conf.cpu())
            conf_time_list.append(conf_test_end - conf_test_start)

        return {
            "y_pred": torch.cat(pred_list),
            "y_true": torch.cat(target_list),
            "eta": torch.cat(conf_list),
            "test_time": np.sum(test_time_list),
            "conf_test_time": np.sum(conf_time_list),
            "test_encode_time": np.sum(test_encode_time_list),
        }


def run_hd(
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
            smiles_train,
            tokenizer=config.tokenizer,
            ngram_order=config.ngram_order,
            num_workers=1,
        )
        test_toks = tokenize_smiles(
            smiles_test,
            tokenizer=config.tokenizer,
            ngram_order=config.ngram_order,
            num_workers=1,
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    result_dict = {"trials": {}}

    for i in range(args.n_trials):
        trial_dict = {}

        # this should force each call of .fit to be different...I think..
        seed_rngs(args.random_state + i)

        model, learning_curve, single_pass_train_time, retrain_time, _ = train_hdc(
            model=model,
            train_dataloader=train_dataloader,
            num_epochs=config.epochs,
            device=config.device,
        )

        trial_dict["hd_learning_curve"] = learning_curve

        # import pdb
        # pdb.set_trace()
        # time test inside of the funcion
        test_dict = test_hdc(model, test_dataloader, device=config.device)

        trial_dict["am"] = {
            0: model.am[0].cpu().numpy(),
            1: model.am[1].cpu().numpy(),
        }  # store the associative memory so it can be loaded up later on

        trial_dict["y_pred"] = test_dict["y_pred"].cpu().numpy()
        trial_dict["eta"] = test_dict["eta"].cpu().numpy().reshape(-1, 1)
        trial_dict["y_true"] = test_dict["y_true"].cpu().numpy()
        trial_dict["single_pass_train_time"] = single_pass_train_time
        trial_dict["test_time"] = test_dict["test_time"]
        trial_dict["conf_test_time"] = test_dict["conf_test_time"]
        trial_dict["train_encode_time"] = test_encode_time
        trial_dict["test_encode_time"] = test_encode_time
        trial_dict["encode_time"] = train_encode_time + test_encode_time

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

        # import pdb
        # pdb.set_trace()
        validate(
            labels=trial_dict["y_true"],
            pred_labels=trial_dict["y_pred"],
            pred_scores=trial_dict["eta"],
        )
        result_dict["trials"][i] = trial_dict
    
    return result_dict


def train_mlp(model, train_dataloader, epochs, device):
    model = model.to(device)

    forward_time = 0.0
    loss_time = 0.0
    backward_time = 0.0
    step = 0

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc=f"training MLP epoch: {epoch}"):
            model.optimizer.zero_grad()

            x, y = batch

            x = x.to(device).float()
            y = y.to(device).reshape(-1).long()

            forward_start = time.time()
            y_ = model(x)
            forward_end = time.time()


            loss_start = time.time()
            loss = model.criterion(y_.reshape(-1, 2), y)
            loss_end = time.time()

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()

            model.optimizer.step()

            step += 1

            forward_time += forward_end - forward_start
            loss_time += loss_end - loss_start
            backward_time += backward_end - backward_start

    return {
        "model": model,
        "train_time": forward_time + loss_time + backward_time,
        "forward_time": forward_time,
        "loss_time": loss_time,
        "backward_time": backward_time,
    }


def val_mlp(model, val_dataloader, device):
    forward_time = 0.0
    loss_time = 0.0
    total_loss = 0.0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"validating MLP"):
            x, y = batch
            targets.append(y)

            x = x.to(device).float()
            y = y.to(device).reshape(-1).long()

            forward_start = time.time()
            y_ = model(x)
            forward_end = time.time()

            preds.append(y_)

            loss_start = time.time()
            loss = model.criterion(y_, y)
            loss_end = time.time()

            total_loss += loss.item()

            forward_time += forward_end - forward_start
            loss_time += loss_end - loss_start

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    # import pdb
    # pdb.set_trace()
    return {
        "y_true": targets.to("cpu"),
        "y_pred": torch.argmax(preds, dim=1).to("cpu"),
        "eta": preds.reshape(-1, 2).to("cpu"),
        "loss": total_loss,
        "forward_time": forward_time,
        "loss_time": loss_time,
    }


def ray_mlp_job(params, train_dataloader, val_dataloader):
    model = MLPClassifier(**params)

    train_dict = train_mlp(
        model=model, train_dataloader=train_dataloader, epochs=5, device=config.device
    )
    model = train_dict["model"]

    val_dict = val_mlp(model=model, val_dataloader=val_dataloader, device=config.device)

    loss = val_dict["loss"] / val_dict["y_pred"].shape[0]
    val_time = val_dict["forward_time"] / val_dict["y_pred"].shape[0]

    tune.report(loss=loss, val_time=val_time)


def run_mlp(train_dataset, test_dataset):
    param_dist = {
        "layer_sizes": tune.choice(
            [
                ((config.input_size, 512), (512, 256), (256, 128), (128, 2)),
                ((config.input_size, 256), (256, 128), (128, 64), (64, 2)),
                ((config.input_size, 128), (128, 64), (64, 32), (32, 2)),
                ((config.input_size, 512), (512, 128), (128, 2)),
                ((config.input_size, 256), (256, 64), (64, 2)),
                ((config.input_size, 128), (128, 32), (32, 2)),
                ((config.input_size, 512), (512, 2)),
                ((config.input_size, 256), (256, 2)),
                ((config.input_size, 128), (128, 2)),
            ]
        ),
        #"lr": tune.choice([1e-3, 1e-2],
        "lr": tune.uniform(1e-5, 1e-1),
        "activation": tune.choice([torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.GELU()]),
        "criterion": tune.choice([torch.nn.NLLLoss()]),
        "optimizer": tune.choice([torch.optim.Adam, torch.optim.SGD]),
    }

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [0.80, 0.20]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
    )


    from ray.tune.schedulers import FIFOScheduler
    # scheduler = FIFOScheduler()


    scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=2,
        brackets=2,
    )



    # from ray.tune.schedulers import PopulationBasedTraining

    # scheduler = PopulationBasedTraining(
        # time_attr='training_iteration',
        # metric='loss',
        # mode='min',
        # perturbation_interval=1,
        # hyperparam_mutations={
            # "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            # "alpha": tune.uniform(0.0, 1.0),
        # }
        # hyperparam_mutations=param_dist
    # )



    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                ray_mlp_job,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
            ),
            # resources={"cpu": 4, "gpu": 1},
            resources={"gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=50,
        ),
        param_space=param_dist,
        run_config=RunConfig(verbose=1),
    )
    results = tuner.fit()

    # import ipdb
    # ipdb.set_trace()

    # get best model then run on full training set for config.num_epochs

    best_params = results.get_best_result("loss", "min").config

    print(f"best MLP params: {best_params}")

    model = MLPClassifier(**best_params)

    result_dict = {"best_params": best_params, "trials": {}}
    for i in range(args.n_trials):
        trial_dict = {}

        train_dict = train_mlp(
            model=model,
            train_dataloader=train_dataloader,
            epochs=config.epochs,
            device=config.device,
        )
        test_dict = val_mlp(
            model=train_dict["model"],
            val_dataloader=test_dataloader,
            device=config.device,
        )

        # trial_dict = {}
        seed = args.random_state + i
        # this should force each call of .fit to be different...I think..
        seed_rngs(seed)
        # collect the best parameters and train on full training set, we capture timings wrt to the optimal configuration
        # construct after seeding the rng

        trial_dict["y_pred"] = test_dict["y_pred"].cpu()
        trial_dict["eta"] = test_dict["eta"].cpu()
        trial_dict["y_true"] = test_dict["y_true"]
        trial_dict["train_time"] = train_dict["train_time"]
        trial_dict["test_time"] = test_dict["forward_time"]
        trial_dict["train_encode_time"] = None
        trial_dict["test_encode_time"] = None
        trial_dict["encode_time"] = None
        trial_dict["train_size"] = len(train_dataset)
        trial_dict["test_size"] = len(test_dataset)

        trial_dict["class_report"] = classification_report(
            y_pred=trial_dict["y_pred"], y_true=trial_dict["y_true"]
        )

        # import pdb
        # pdb.set_trace()
        try:
            trial_dict["roc-auc"] = roc_auc_score(
                y_score=trial_dict["eta"][:, 1], y_true=trial_dict["y_true"]
            )

        except ValueError as e:
            trial_dict["roc-auc"] = None
            print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        print(trial_dict["class_report"])
        print(f"roc-auc {trial_dict['roc-auc']}")

        # enrichment metrics
        # import pdb
        # pdb.set_trace()
        validate(
            labels=trial_dict["y_true"],
            pred_labels=trial_dict["y_pred"],
            pred_scores=trial_dict["eta"][:, 1],
        )
        result_dict["trials"][i] = trial_dict

    return result_dict


def main(
    model,
    train_dataset,
    test_dataset,
):
    if config.model in ["molehd", "selfies", "ecfp", "rp"]:
        result_dict = run_hd(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
    elif config.model in ["mlp"]:
        result_dict = run_mlp(train_dataset=train_dataset, test_dataset=test_dataset)
    else:
        raise NotImplementedError

    return result_dict


def driver():
    train_dataset, test_dataset = None, None
    # todo: ngram order and tokenizer only apply to some models, don't need to have in the exp_name
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

    result_dict = None
    roc_values = [] # some datasets contain multiple targets, store these values then print at end
    if args.dataset == "bbbp":
        smiles_col = "rdkitSmiles"
        label_col = "p_np"
        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv"
        )

        split_path = Path(
            f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        split_df = compute_splits(
            split_path=split_path,
            random_state=args.random_state,
            split_type=args.split_type,
            df=df,
            smiles_col="rdkitSmiles",
        )

        x_train, x_test, y_train, y_test = None, None, None, None
        smiles_train, smiles_test = None, None
        if config.embedding == "molformer":
            dataset = MolFormerDataset(
                path=f"{SCRATCH_DIR}/molformer_embeddings/bbbp_molformer_embeddings.pt",
                split_df=split_df,
                smiles_col="rdkitSmiles",
            )

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            smiles_train, smiles_test = dataset.smiles_train, dataset.smiles_test
            train_dataset = TensorDataset(
                torch.from_numpy(x_train).int(), torch.from_numpy(y_train).int()
            )
            test_dataset = TensorDataset(
                torch.from_numpy(x_test).int(), torch.from_numpy(y_test).int()
            )

        elif config.embedding == "ecfp":
            dataset = ECFPDataset(
                path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv",
                smiles_col=smiles_col,
                label_col=label_col,
                #   split_df=df,
                smiles=df[smiles_col],
                split_df=split_df,
                split_type=args.split_type,
                random_state=args.random_state,
                ecfp_length=config.input_size,
                ecfp_radius=config.ecfp_radius,
                labels=df["p_np"].values,
            )
            smiles_train = dataset.smiles_train
            smiles_test = dataset.smiles_test
            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            train_dataset = TensorDataset(x_train.int(), y_train.int())
            test_dataset = TensorDataset(x_test.int(), y_test.int())

        elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
            train_smiles = split_df[smiles_col][
                split_df[split_df["split"] == "train"]["index"]
            ]
            test_smiles = split_df[smiles_col][
                split_df[split_df["split"] == "test"]["index"]
            ]

            train_labels = split_df[label_col][
                split_df[split_df["split"] == "train"]["index"]
            ]
            test_labels = split_df[label_col][
                split_df[split_df["split"] == "test"]["index"]
            ]

            train_dataset = SMILESDataset(
                smiles=train_smiles,
                labels=train_labels,
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                num_workers=16,
                device=device,
            )
            # use the item_memory generated by the train_dataset as a seed for the test, then update both?
            test_dataset = SMILESDataset(
                smiles=test_smiles,
                labels=test_labels,
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                item_mem=train_dataset.item_mem,
                num_workers=1,
                device=device,
            )

            train_dataset.item_mem = test_dataset.item_mem
            model.item_mem = train_dataset.item_mem
        else:
            raise NotImplementedError

        output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{args.split_type}.{args.random_state}.pkl"
        )
        if output_file.exists():
            print(f"{output_file} exists. loading previous result.")
            result_dict = torch.load(output_file)
        else:
            result_dict = main(
                model=model, train_dataset=train_dataset, test_dataset=test_dataset
            )

            result_dict["smiles_train"] = smiles_train
            result_dict["smiles_test"] = smiles_test
            result_dict["y_train"] = y_train
            result_dict["y_test"] = y_test

            result_dict["args"] = config
            torch.save(result_dict, output_file)
            print(f"done. output file {output_file}")

        roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))

    elif args.dataset == "sider":
        # import pdb
        # pdb.set_trace()
        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/sider.csv"
        )

        split_path = Path(
            f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        # import pdb
        # pdb.set_trace()

        # roc_list = []

        split_df = compute_splits(
            label_col=df.columns[0], # the split shouldnot depend on the labels if using scaffold iirc https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
            split_path=split_path,
            random_state=args.random_state,
            split_type=args.split_type,
            df=df,
            smiles_col="smiles",
        )

        label_list = [
            x for x in split_df.columns if x not in ["split", "smiles", "index"]
        ]
        for task_idx, label in enumerate(label_list):
            output_file = Path(
                f"{output_result_dir}/{exp_name}.{args.dataset}-{task_idx}.{args.random_state}.pkl"
            )

            if config.embedding == "molformer":
                dataset = MolFormerDataset(
                    path=f"{SCRATCH_DIR}/molformer_embeddings/sider-{task_idx}_molformer_embeddings.pt",
                    split_df=split_df,
                    smiles_col="smiles",
                )

                x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                smiles_train, smiles_test = dataset.smiles_train, dataset.smiles_test
                train_dataset = TensorDataset(
                    x_train, y_train
                )

                test_dataset = TensorDataset(
                    x_test, y_test
                )
            if config.embedding == "ecfp":
                dataset = ECFPDataset(
                    path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/sider.csv",
                    smiles_col="smiles",
                    label_col=label,
                    smiles=df["smiles"],
                    split_df=split_df,
                    split_type=args.split_type,
                    random_state=args.random_state,
                    ecfp_length=config.input_size,
                    ecfp_radius=config.ecfp_radius,
                    labels=df[label].values,
                )
                smiles_train = dataset.smiles_train
                smiles_test = dataset.smiles_test
                x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                train_dataset = TensorDataset(x_train, y_train)

                test_dataset = TensorDataset(x_test, y_test)

            if output_file.exists():
                print(f"output_file: {output_file} exists. skipping.")
                result_dict = torch.load(output_file)

            else:
                result_dict = main(
                    model=model, train_dataset=train_dataset, test_dataset=test_dataset
                )

                result_dict["smiles_train"] = smiles_train
                result_dict["smiles_test"] = smiles_test

                result_dict["y_train"] = y_train
                result_dict["y_test"] = y_test

                result_dict["args"] = config
                torch.save(result_dict, output_file)

                print(f"done. output file {output_file}")
            # average over the trials dimension
            roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))
    elif args.dataset == "clintox":
        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/clintox-cleaned.csv"
        )

        # some of the smiles (6) for clintox dataset are not able to be converted to fingerprints, so skip them for all cases

        split_path = Path(
            f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        # import pdb
        # pdb.set_trace()
        split_df = compute_splits(
            split_path=split_path,
            random_state=args.random_state,
            split_type=args.split_type,
            df=df,
            smiles_col="smiles",
            label_col="CT_TOX",
        )

        if config.embedding == "molformer":
            dataset = MolFormerDataset(
                path=f"{SCRATCH_DIR}/molformer_embeddings/clintox_molformer_embeddings.pt",
                split_df=split_df,
                smiles_col="smiles",
            )

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()

            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)

        elif config.embedding == "ecfp":
            dataset = ECFPDataset(
                path="/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/clintox-cleaned.csv",
                smiles_col="smiles",
                label_col="FDA_APPROVED",
                #   split_df=df,
                smiles=df["smiles"],
                split_df=split_df,
                split_type=args.split_type,
                random_state=args.random_state,
                ecfp_length=config.input_size,
                ecfp_radius=config.ecfp_radius,
                labels=df["FDA_APPROVED"].values,
            )

            x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
        elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
            train_smiles = split_df[smiles_col][
                split_df[split_df["split"] == "train"]["index"]
            ]
            test_smiles = split_df[smiles_col][
                split_df[split_df["split"] == "test"]["index"]
            ]

            train_labels = split_df[label_col][
                split_df[split_df["split"] == "train"]["index"]
            ]
            test_labels = split_df[label_col][
                split_df[split_df["split"] == "test"]["index"]
            ]

            train_dataset = SMILESDataset(
                smiles=train_smiles,
                labels=train_labels,
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                num_workers=16,
                device=device,
            )
            # use the item_memory generated by the train_dataset as a seed for the test, then update both?
            test_dataset = SMILESDataset(
                smiles=test_smiles,
                labels=test_labels,
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                item_mem=train_dataset.item_mem,
                num_workers=1,
                device=device,
            )

            train_dataset.item_mem = test_dataset.item_mem
            model.item_mem = train_dataset.item_mem
        else:
            raise NotImplementedError

        output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}.{args.split_type}.{args.random_state}.pkl"
        )

        if output_file.exists():
            print(f"output_file: {output_file} exists. skipping.")
            result_dict = torch.load(output_file)

        else:
            result_dict = main(
                model=model, train_dataset=train_dataset, test_dataset=test_dataset
            )

            result_dict["smiles_train"] = dataset.smiles_train
            result_dict["smiles_test"] = dataset.smiles_test
            result_dict["y_train"] = y_train
            result_dict["y_test"] = y_test

            result_dict["args"] = config
            torch.save(result_dict, output_file)
            print(f"done. output file: {output_file}")

        roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))
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
                result_dict = torch.load(output_file) 
            else:
                print(f"processing {dude_smiles_path}...")

                smiles_df = pd.read_csv(dude_smiles_path)

                dude_split_path = None
                if args.split_type == "random":
                    dude_split_path = dude_smiles_path.with_name(
                        dude_smiles_path.stem
                        + "_with_base_rdkit_smiles_train_valid_test_random_random.csv"
                    )

                split_df = pd.read_csv(dude_split_path)

                df = pd.merge(smiles_df, split_df, left_on="id", right_on="cmpd_id")
                df["split"] = df["subset"]
                df["index"] = df.index
                df["decoy"] = df["decoy"].apply(lambda x:  1- x)
                # df = df.sample(frac=1.0)

                smiles_col = "smiles"
                label_col = "decoy"

                df[label_col] = df[label_col].astype(int).values



                if config.embedding == "ecfp":
                    dataset = ECFPDataset(
                        path=dude_smiles_path,
                        smiles_col=smiles_col,
                        label_col=label_col,
                        smiles=df[smiles_col],
                        split_df=df,
                        split_type=args.split_type,
                        random_state=args.random_state,
                        ecfp_length=config.input_size,
                        ecfp_radius=config.ecfp_radius,
                        labels=df[label_col].values,
                    )

                    x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                    train_dataset = TensorDataset(x_train, y_train)
                    test_dataset = TensorDataset(x_test, y_test)

                elif config.embedding == "molformer":
                    train_dataset = MolFormerDataset(
                        path=f"{SCRATCH_DIR}/molformer_embeddings/dude-{target_name}_random_train_molformer_embeddings.pt",
                        split_df=df,
                        smiles_col="smiles",
                    )
                    test_dataset = MolFormerDataset(
                        path=f"{SCRATCH_DIR}/molformer_embeddings/dude-{target_name}_random_test_molformer_embeddings.pt",
                        split_df=df,
                        smiles_col="smiles",
                    )

                    x_train = train_dataset.x
                    y_train = train_dataset.y
                    x_test = test_dataset.x
                    y_test = test_dataset.y


                elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
                    train_smiles = df[smiles_col][
                        df[df["split"] == "train"]["index"]
                    ]
                    test_smiles = df[smiles_col][
                        df[df["split"] == "test"]["index"]
                    ]

                    train_labels = (df[label_col][
                        df[df["split"] == "train"]["index"]
                    ]).values
                    test_labels = (df[label_col][
                        df[df["split"] == "test"]["index"]
                    ]).values

                    train_dataset = SMILESDataset(
                        smiles=train_smiles,
                        labels=train_labels,
                        D=config.D,
                        tokenizer=config.embedding,
                        ngram_order=config.ngram_order,
                        num_workers=16,
                        device=device,
                    )
                    # use the item_memory generated by the train_dataset as a seed for the test, then update both?
                    test_dataset = SMILESDataset(
                        smiles=test_smiles,
                        labels=test_labels,
                        D=config.D,
                        tokenizer=config.embedding,
                        ngram_order=config.ngram_order,
                        item_mem=train_dataset.item_mem,
                        num_workers=1,
                        device=device,
                    )

                    train_dataset.item_mem = test_dataset.item_mem
                    model.item_mem = train_dataset.item_mem


                    y_test = test_dataset.labels
                    # import pdb
                    # pdb.set_trace()



                # import pdb
                # pdb.set_trace()
                if config.embedding in ["ecfp"]:
                    x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                    if isinstance(x_train, np.ndarray):
                        x_train, x_test, y_train, y_test = (
                            torch.from_numpy(x_train),
                            torch.from_numpy(x_test),
                            torch.from_numpy(y_train),
                            torch.from_numpy(y_test),
                        )
                    train_dataset = TensorDataset(x_train, y_train)
                    test_dataset = TensorDataset(x_test, y_test)



                result_dict = main(
                    model=model, train_dataset=train_dataset, test_dataset=test_dataset
                )
                # result_dict["y_train"] = y_train
                result_dict["y_test"] = y_test
                result_dict["args"] = config
                torch.save(result_dict, output_file)

                # import pdb
                # pdb.set_trace()
                print(f"done. output file: {output_file}")

            roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))

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
                f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name}.{args.random_state}.pkl"
            )

            if output_file.exists():
                # don't recompute if it's already been calculated
                print(
                    f"output file: {output_file} for input file {lit_pcba_path} already exists. moving on to next target.."
                )

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

                if config.embedding == "ecfp":
                    dataset = ECFPDataset(
                        path=lit_pcba_path / Path("full_smiles_with_labels.csv"),
                        smiles_col="smiles",
                        label_col="label",
                        #   split_df=df,
                        smiles=df["smiles"],
                        split_df=df,
                        split_type=args.split_type,
                        random_state=args.random_state,
                        ecfp_length=config.input_size,
                        ecfp_radius=config.ecfp_radius,
                        labels=df["label"].astype(int).values,
                    )

                elif config.embedding == "molformer":
                    dataset = MolFormerDataset(
                        path=f"{SCRATCH_DIR}/molformer_embeddings/lit-pcba-{target_name}_molformer_embeddings.pt",
                        split_df=df,
                        smiles_col="smiles",
                    )

                elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
                    train_smiles = df["smiles"][df[df["split"] == "train"]["index"]]
                    test_smiles = df["smiles"][df[df["split"] == "test"]["index"]]

                    train_labels = df["label"][df[df["split"] == "train"]["index"]]
                    test_labels = df["label"][df[df["split"] == "test"]["index"]]

                    train_dataset = SMILESDataset(
                        smiles=train_smiles,
                        labels=train_labels,
                        D=config.D,
                        tokenizer=config.embedding,
                        ngram_order=config.ngram_order,
                        num_workers=32,
                        device=device,
                    )
                    # use the item_memory generated by the train_dataset as a seed for the test, then update both?
                    test_dataset = SMILESDataset(
                        smiles=test_smiles,
                        labels=test_labels,
                        D=config.D,
                        tokenizer=config.embedding,
                        ngram_order=config.ngram_order,
                        item_mem=train_dataset.item_mem,
                        num_workers=1,
                        device=device,
                    )

                    train_dataset.item_mem = test_dataset.item_mem
                    model.item_mem = train_dataset.item_mem
                else:
                    raise NotImplementedError

                # this only applies to the molformer and ecfp input cases

                if config.embedding in ["molformer", "ecfp"]:
                    x_train, x_test, y_train, y_test = dataset.get_train_test_splits()
                    if isinstance(x_train, np.ndarray):
                        x_train, x_test, y_train, y_test = (
                            torch.from_numpy(x_train),
                            torch.from_numpy(x_test),
                            torch.from_numpy(y_train),
                            torch.from_numpy(y_test),
                        )
                    train_dataset = TensorDataset(x_train, y_train)
                    test_dataset = TensorDataset(x_test, y_test)

                if output_file.exists():
                    with open(output_file, "rb") as handle:
                        result_dict = pickle.load(handle)

                else:
                    # with torch.profiler.profile(
                        # activities=[
                            # torch.profiler.ProfilerActivity.CUDA,
                        # ],
                        # with_stack=True,
                        # profile_memory=True,
                    # ) as p:
                    result_dict = main(
                        model=model,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                    )
                    # print(
                        # p.key_averages().table(
                            # sort_by="self_cuda_time_total", row_limit=-1
                        # )
                    # )

                    # p.export_chrome_trace(
                        # f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name}.{args.random_state}.json"
                    # )
                    # p.export_stacks(
                        # f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name}.{args.random_state}_cuda_stacks.txt",
                        # "self_cuda_time_total",
                    # )

                    if config.model == "molehd":
                        result_dict["y_train"] = train_dataset.labels.values
                        result_dict["y_test"] = test_dataset.labels.values
                    else:
                        result_dict["y_train"] = (train_dataset.tensors[1]).numpy()
                        result_dict["y_test"] = (test_dataset.tensors[1]).numpy()

                    result_dict["smiles_train"] = dataset.smiles_train
                    result_dict["smiles_test"] = dataset.smiles_test

                    result_dict["args"] = config

                    torch.save(result_dict, output_file)

                roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))
 
                print(f"done. output file: {output_file}")

    print(f"Average ROC-AUC is {np.mean(roc_values)} +/- ({np.std(roc_values)})")


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

    exp_name = f"{Path(args.config).stem}"

    driver()

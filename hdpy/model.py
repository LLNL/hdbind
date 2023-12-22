################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import torch
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
# from hdpy.molehd.encode import tokenize_smiles
from hdpy.utils import collate_list_fn, binarize, seed_rngs
from sklearn.metrics import classification_report, roc_auc_score
from ray import tune
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.air import Checkpoint, RunConfig
from ray import train, tune
from hdpy.metrics import validate

class HDModel(nn.Module):
    def __init__(self, D: int):
        super(HDModel, self).__init__()
        self.am = None
        self.D = D

    def build_item_memory(self, x_train, train_labels):
        raise NotImplementedError("Please implement this function in a subclass")

    def build_am(self, dataset_hvs, labels):
        # raise NotImplementedError("Please implement this function in a subclass")

        self.am = {}

        for hv, label in zip(dataset_hvs, labels):
            if int(label) not in self.am.keys():
                self.am[int(label)] = hv
            else:
                self.am[int(label)] += hv

    def update_am(self, dataset_hvs, labels):
        # this avoids creating a new associative memory and instead just updates the existing one...not sure why we need two functions so that should be updated at some point
        if self.am is None:
            self.am = {}
        for hv, label in zip(dataset_hvs, labels):
            if int(label) not in self.am.keys():
                self.am[int(label)] = hv
            else:
                self.am[int(label)] += hv

    def encode(self, x):
        raise NotImplementedError("Please implement this function in a subclass")

    def predict(self, enc_hvs):
        start = time.time()
        # preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().cuda().float(), self.am.clone().cuda().float()), dim=1)
        # preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().float(), self.am.clone().float()), dim=1)
        preds = torch.argmax(
            torchmetrics.functional.pairwise_cosine_similarity(
                enc_hvs.float(), self.am.float()
            ),
            dim=1,
        )
        end = time.time()
        # print((end-start)/enc_hvs.shape[0])
        return preds

    # def forward(self, x, encode_only=False):

    # if encode_only:
    # out = self.forward(x)
    # out = self.predict(x)
    # return out

    def compute_confidence(self, dataset_hvs):
        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order

        # this torchmetrics function potentially edits in place so we make a clone
        # moving the self.am to cuda memory repeatedly seems wasteful
        sims = torchmetrics.functional.pairwise_cosine_similarity(
            dataset_hvs.clone().float().cuda(), self.am.clone().float().cuda()
        )

        eta = (sims[:, 1] - sims[:, 0]) * (1 / 4)
        eta = torch.add(eta, (1 / 2))
        return eta.reshape(-1)

    def retrain(self, dataset_hvs, labels, return_mistake_count=False, lr=1.0):
        # should do this in parallel instead of the sequential? or can we define two functions and possibly combine the two? i.e. use the parallel version most of the time and then periodically update with the sequential version?

        shuffle_idx = torch.randperm(dataset_hvs.size()[0])
        dataset_hvs = dataset_hvs[shuffle_idx].int()
        labels = labels[shuffle_idx].int()

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order

        mistakes = 0

        for hv, label in tqdm(zip(dataset_hvs, labels), total=len(dataset_hvs)):
            out = int(
                torch.argmax(torch.nn.CosineSimilarity()(hv.float(), self.am.float()))
            )

            if out == int(label):
                pass
            else:
                mistakes += 1
                self.am[out] -= (lr * hv).int()
                self.am[int(label)] += (lr * hv).int()

        if return_mistake_count:
            return mistakes

    def fit(self, x_train, y_train, num_epochs, lr=1.0):
        for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
            self.train_step(train_features=x_train, train_labels=y_train, lr=lr)


"""
import pytorch_lightning as pl
class RPHDLightningModule(pl.LightningModule):

    def __init__(self, D:int, input_size:int, num_classes:int, init_type="uniform"):

        super()
        self.D = D
        self.input_size=input_size
        self.num_classes=num_classes
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        if init_type == "uniform":
            init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float()*2-1
            self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

            self.init_class_hvs = torch.zeros(num_classes, D).float()


    def forward(self, x):
                
        hv = self.rp_layer(x.float())
        hv = torch.where(hv>0, 1.0, -1.0)
        return hv
"""


class RPEncoder(HDModel):
    def __init__(self, input_size: int, D: int, num_classes: int):
        super(RPEncoder, self).__init__(D=D)
        self.rp_layer = nn.Linear(input_size, D, bias=False)
        init_rp_mat = (
            torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float() * 2 - 1
        )
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float()

        self.am = torch.zeros(2, self.D, dtype=int)
        self.name = "rp"

    def encode(self, x):
        hv = self.rp_layer(x.float())
        hv = torch.where(hv > 0, 1.0, -1.0).int()
        return hv

    def forward(self, x):
        hv = self.rp_layer(x.float())
        hv = torch.where(hv > 0, 1.0, -1.0).int()

        return hv


class TokenEncoder(HDModel):
    def __init__(self, D: int, num_classes: int, item_mem=dict):
        super(TokenEncoder, self).__init__(D=D)

        self.D = D
        self.num_classes = num_classes
        self.item_mem = item_mem
        self.am = torch.zeros(2, self.D, dtype=int)
        self.name = "molehd"

    def encode(self, tokens: list):
        # tokens is a list of tokens that we will map to item_mem token hvs and produce the smiles hv
        hv = torch.zeros(1, self.D).int()

        batch_tokens = [
            torch.roll(self.item_mem[token], idx).reshape(1, -1)
            for idx, token in enumerate(tokens)
        ]

        hv = torch.vstack(batch_tokens).sum(dim=0).reshape(1, -1)

        # binarize
        hv = torch.where(hv > 0, hv, -1).int()
        hv = torch.where(hv <= 0, hv, 1).int()

        return hv

    def forward(self, x):
        return super().forward(x).cpu()

    def encode_batch(self, token_list: list):
        return torch.cat([self.encode(z) for z in token_list])


class HD_Kron_Classification(HDModel):
    def __init__(self, Kron_shape, input_size, D, num_classes, binary=True):
        super(HD_Kron_Classification, self).__init__()
        self.Kron_shape = Kron_shape
        self.D, self.F = D, input_size
        self.binary = binary

        self.Kron_1 = nn.Linear(Kron_shape[1], Kron_shape[0], bias=False)
        self.Kron_2 = nn.Linear(
            self.F // Kron_shape[1], self.D // Kron_shape[0], bias=False
        )
        self.init_class_hvs = torch.zeros(num_classes, self.D).float().cuda()

        if binary:
            init_rp_mat = (
                torch.bernoulli(
                    torch.tensor([[0.5] * Kron_shape[1]] * Kron_shape[0])
                ).float()
                * 2
                - 1
            )
            self.Kron_1.weight = nn.parameter.Parameter(
                init_rp_mat, requires_grad=False
            )
            init_rp_mat = (
                torch.bernoulli(
                    torch.tensor(
                        [[0.5] * (self.F // Kron_shape[1])] * (self.D // Kron_shape[0])
                    )
                ).float()
                * 2
                - 1
            )
            self.Kron_2.weight = nn.parameter.Parameter(
                init_rp_mat, requires_grad=False
            )

    def RP_encoding(self, x):
        x = x.view(x.size()[0], self.F // self.Kron_shape[1], self.Kron_shape[1])
        out = self.Kron_1(x)
        out = self.Kron_2(out.permute(0, 2, 1))
        out = out.view(out.size()[0], -1)

        if self.binary:
            out = torch.where(out > 0, 1.0, -1.0)
        return out

    def init_class(self, x_train, train_labels):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[train_labels[i]] += out[i]
        self.am = self.init_class_hvs

        self.am = binarize(self.am)


class HD_Level_Classification(HDModel):
    def __init__(self, input_size, D, num_classes, quan_level=8):
        super(HD_Level_Classification, self).__init__()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        # self.
        self.quantize_scale = 1.0 / quan_level

        density = 0.5

        init_rp_mat = torch.bernoulli(
            torch.tensor([[density] * input_size] * D)
        ).float()
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()

    def quantize(self, x):
        return torch.fake_quantize_per_tensor_affine(
            x, scale=self.quantize_scale, zero_point=0, quant_min=0, quant_max=3
        )

    # def encoding(self, x):
    #    out = self.rp_layer(x)
    # out = torch.where(out>0, 1.0, -1.0)
    #    return out

    def encode(self, x):
        return self.RP_encoding(x)

    def RP_encoding(self, x):
        # ipdb.set_trace()
        out = self.rp_layer(x)
        out = torch.where(out > 0, 1.0, -1.0)
        return out

    def init_class(self, x_train, train_labels):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[train_labels[i]] += out[i]
        # self.am = nn.parameter.Parameter(self.init_class_hvs, requires_grad=True)
        self.am = self.init_class_hvs

    def forward(self, x):
        out = self.level_encoding(x)
        out = nn.CosineSimilarity()(class_hvs=self.am, enc_hv=out)
        return out


# BENCHMARK MODELS


# Fully connected neural network with one hidden layer
class MLPClassifier(nn.Module):
    def __init__(self, layer_sizes, lr, activation, criterion, optimizer):
        super(MLPClassifier, self).__init__()
        self.activation = activation

        self.fc_layers = torch.nn.Sequential()
        for idx, (input_size, output_size) in enumerate(layer_sizes):
            if idx < len(layer_sizes) - 1:
                self.fc_layers.append(nn.Linear(input_size, output_size))
                self.fc_layers.append(self.activation)
            else:
                self.fc_layers.append(nn.Linear(input_size, output_size))

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        out = self.fc_layers(x)
        out = torch.nn.LogSoftmax(dim=1)(out)
        return out


from sklearn.neighbors import KNeighborsClassifier


class kNN(nn.Module):
    def __init__(self, model_type):
        self.model = KNeighborsClassifier(n_neighbors=1, metric=model_type.lower())

    def forward(self, x):
        return self.model.predict(x)

    def fit(self, features, labels, num_epochs):
        self.model.fit(features.cpu(), labels.cpu())

    def predict(self, features):
        return self.model.predict(features.cpu())





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
                # import pdb
                # pdb.set_trace()
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

def run_mlp(config,batch_size, num_workers,n_trials, random_state, train_dataset, test_dataset):
    param_dist = {
        "layer_sizes": tune.choice(
            [
                ((config.ecfp_length, 512), (512, 256), (256, 128), (128, 2)),
                ((config.ecfp_length, 256), (256, 128), (128, 64), (64, 2)),
                ((config.ecfp_length, 128), (128, 64), (64, 32), (32, 2)),
                ((config.ecfp_length, 512), (512, 128), (128, 2)),
                ((config.ecfp_length, 256), (256, 64), (64, 2)),
                ((config.ecfp_length, 128), (128, 32), (32, 2)),
                ((config.ecfp_length, 512), (512, 2)),
                ((config.ecfp_length, 256), (256, 2)),
                ((config.ecfp_length, 128), (128, 2)),
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
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
    )




    scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=2,
        brackets=2,
    )



    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                ray_mlp_job,
                device=config.device,
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
    for i in range(n_trials):
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
        seed = random_state + i
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

def run_hd(
    model,
    config,
    batch_size,
    num_workers,
    n_trials,
    random_state,
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
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        persistent_workers=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    result_dict = {"trials": {}}

    for i in range(n_trials):
        trial_dict = {}

        # this should force each call of .fit to be different...I think..
        seed_rngs(random_state + i)

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


def ray_mlp_job(params, device, train_dataloader, val_dataloader):
    model = MLPClassifier(**params)

    train_dict = train_mlp(
        model=model, train_dataloader=train_dataloader, epochs=5, device=device
    )
    model = train_dict["model"]

    val_dict = val_mlp(model=model, val_dataloader=val_dataloader, device=device)

    loss = val_dict["loss"] / val_dict["y_pred"].shape[0]
    val_time = val_dict["forward_time"] / val_dict["y_pred"].shape[0]

    tune.report(loss=loss, val_time=val_time)


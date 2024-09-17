################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import torch
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
from torch.utils.data import DataLoader
import numpy as np
import time
from hdpy.utils import collate_list_fn, binarize_ as binarize, bipolarize, seed_rngs
from hdpy.molehd import tokenize_smiles
from sklearn.metrics import classification_report, roc_auc_score

try:
    from ray.tune.schedulers import ASHAScheduler
    from ray.air import RunConfig
    from ray import tune
except ModuleNotFoundError as e:
    print("ray not available. MLP models not available in this env.")
from hdpy.metrics import (
    validate,
    compute_enrichment_factor,
    matrix_tanimoto_similarity,
    pairwise_hamming_distance,
)


class HDModel(nn.Module):
    def __init__(
        self,
        D: int,
        name=None,
        sim_metric="cosine",
        binarize_am=False,
        bipolarize_am=False,
        binarize_hv=False,
        bipolarize_hv=False,
    ):
        super(HDModel, self).__init__()
        if name:
            self.name = name

        assert not (binarize_am and bipolarize_am)  # only choose 1 if you choose at all
        assert not (binarize_hv and bipolarize_hv)  # only choose 1 if you choose at all

        self.am = None
        self.binarize_am = binarize_am
        self.bipolarize_am = bipolarize_am
        self.binarize_hv = binarize_hv
        self.bipolarize_hv = bipolarize_hv

        self.D = D
        self.sim_metric = sim_metric

    def build_item_memory(self, x_train, train_labels):
        raise NotImplementedError("Please implement this function in a subclass")

    def build_am(self, dataset_hvs, labels):
        if self.binarize_am:
            self.am = binarize(self.am)
        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            dataset_hvs = binarize(dataset_hvs)
        if self.bipolarize_am:
            dataset_hvs = bipolarize(dataset_hvs)

        self.am = torch.zeros(2, self.D, dtype=int)

        # print(type(dataset_hvs))

        for hv, label in zip(dataset_hvs, labels):
            if int(label) not in self.am.keys():
                print(dir(self.am))
                self.am[int(label)] = hv.int()
            else:
                self.am[int(label)] += hv.int()

        if self.binarize_am:
            self.am = binarize(self.am)
        if self.bipolarize_am:
            self.am = bipolarize(self.am)

    def update_am(self, hvs, labels):
        if self.binarize_am:
            self.am = binarize(self.am)
        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            hvs = binarize(hvs)
        if self.bipolarize_am:
            hvs = bipolarize(hvs)

        # this avoids creating a new associative memory and instead just updates the existing one...not sure why we need two functions so that should be updated at some point
        if self.am is None:
            self.am = {}
        for hv, label in zip(hvs, labels):
            if int(label) not in self.am.keys():
                print(dir(self.am))
                self.am[int(label)] = hv
            else:
                self.am[int(label)] += hv

        if self.binarize_am:
            self.am = binarize(self.am)
        if self.bipolarize_am:
            self.am = bipolarize(self.am)

    def encode(self, x):
        raise NotImplementedError("Please implement this function in a subclass")

    def predict(self, hvs, return_time=False):
        # import pdb
        # pdb.set_trace()
        if self.binarize_am:
            self.am = binarize(self.am)

        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            hvs = binarize(hvs)
        if self.bipolarize_am:
            hvs = bipolarize(hvs)

        am = self.am
        am = am.float()

        predict_time_cpu_sum = 0
        predict_time_cuda_sum = 0
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        sim_func = None
        if self.sim_metric == "cosine":
            sim_func = torchmetrics.functional.pairwise_cosine_similarity
            # starter.record()
            # preds = torch.argmax(
            # torchmetrics.functional.pairwise_cosine_similarity(
            # enc_hvs, am
            # ),
            # dim=1,
            # )
            # ender.record()
            # torch.cuda.synchronize()

            # convert elapsed time in milliseconds to seconds
            # total_time = starter.elapsed_time(ender) / 1000

        elif self.sim_metric == "tanimoto":
            sim_func = matrix_tanimoto_similarity
            # predict_time_cpu_start = time.perf_counter()
            # starter.record()
            # preds = torch.argmax(
            # matrix_tanimoto_similarity(enc_hvs, am),
            # dim=1,
            # )

            # ender.record()
            # torch.cuda.synchronize()

            # convert elapsed time in milliseconds to seconds
            # predict_time_cuda_batch = starter.elapsed_time(ender) / 1000

            # predict_time_cpu_end = time.perf_counter()
            # predict_time_cpu_sum += (predict_time_cpu_end)

        elif self.sim_metric == "hamming":
            sim_func = pairwise_hamming_distance
            # predict_time_cpu_start = time.perf_counter()
            # starter.record()
            # preds = torch.argmax(
            # pairwise_hamming_distance(enc_hvs, am),
            # dim=1,
            # )
            # ender.record()
            # torch.cuda.synchronize()

            # predict_time_cuda_batch = starter.elapsed_time(ender) / 1000

            # predict_time_cpu_end = time.perf_counter()
            # predict_time_cpu_sum += (predict_time_cpu_end)

        else:
            raise NotImplementedError(f"{self.sim_metric} not implemented.")

        predict_time_cpu_start = time.perf_counter()
        starter.record()

        preds = torch.argmax(
            sim_func(hvs, am),
            dim=1,
        )
        ender.record()
        torch.cuda.synchronize()

        predict_time_cuda_sum = starter.elapsed_time(ender) / 1000

        predict_time_cpu_end = time.perf_counter()
        predict_time_cpu_sum = (
            predict_time_cpu_end - predict_time_cpu_start
        ) - predict_time_cuda_sum

        # if self.binarize_am:
        # self.am = binarize(self.am)

        if return_time:
            return preds, predict_time_cpu_sum, predict_time_cuda_sum
        else:
            return preds

    def compute_confidence(self, hvs, return_time=False):
        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order

        # this torchmetrics function potentially edits in place so we make a clone
        # moving the self.am to cuda memory repeatedly seems wasteful

        if self.binarize_am:
            self.am = binarize(self.am)

        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            hvs = binarize(hvs)
        if self.bipolarize_am:
            hvs = bipolarize(hvs)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        starter.record()
        sims = torchmetrics.functional.pairwise_cosine_similarity(
            hvs.clone().float().cuda(), self.am.clone().float().cuda()
        )
        eta = (sims[:, 1] - sims[:, 0]) * (1 / 4)
        eta = torch.add(eta, (1 / 2)).reshape(-1)
        ender.record()

        # convert milliseconds to seconds
        total_time = starter.elapsed_time(ender) / 1000

        if return_time:
            return eta, total_time
        else:
            return eta

    """
    def retrain(self, dataset_hvs, labels, return_mistake_count=False, lr=1.0):
        # should do this in parallel instead of the sequential? or can we define two functions and possibly combine the two? i.e. use the parallel version most of the time and then periodically update with the sequential version?

        shuffle_idx = torch.randperm(dataset_hvs.size()[0])
        dataset_hvs = dataset_hvs[shuffle_idx].int()
        labels = labels[shuffle_idx].int()

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order

        mistakes = 0

        for hv, label in tqdm(zip(dataset_hvs, labels), total=len(dataset_hvs)):
            # out = int(
                # torch.argmax(torch.nn.CosineSimilarity()(hv.float(), self.am.float()))
            # )
            # out = int(
                # torch.argmax(torch.nn.CosineSimilarity()(hv.float(), self.am.float()))
            # )


            if out == int(label):
                pass
            else:
                mistakes += 1
                self.am[out] -= (lr * hv).int()
                self.am[int(label)] += (lr * hv).int()

        if return_mistake_count:
            return mistakes
    """

    def fit(self, x_train, y_train, num_epochs, lr=1.0):
        if self.binarize_am:
            self.am = binarize(self.am)

        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            hvs = binarize(hvs)
        if self.bipolarize_am:
            hvs = bipolarize(hvs)

        for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
            self.train_step(train_features=x_train, train_labels=y_train, lr=lr)

        if self.binarize_am:
            self.am = binarize(self.am)

        if self.bipolarize_am:
            self.am = bipolarize(self.am)

        if self.binarize_hv:
            hvs = binarize(hvs)
        if self.bipolarize_am:
            hvs = bipolarize(hvs)


class RPEncoder(HDModel):
    def __init__(
        self,
        input_size: int,
        D: int,
        num_classes: int,
        sim_metric: str,
        binarize_am=False,
        bipolarize_am=False,
        binarize_hv=False,
        bipolarize_hv=True,
    ):
        super(RPEncoder, self).__init__(
            D=D,
            sim_metric=sim_metric,
            binarize_am=binarize_am,
            bipolarize_am=bipolarize_am,
            binarize_hv=binarize_hv,
            bipolarize_hv=bipolarize_hv,
        )

        self.rp_layer = nn.Linear(input_size, D, bias=False).float()
        init_rp_mat = (
            torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float() * 2 - 1
        )
        self.rp_layer.weight = nn.parameter.Parameter(
            init_rp_mat, requires_grad=False
        ).float()

        self.init_class_hvs = torch.zeros(num_classes, D).float()

        # self.am = torch.zeros(2, self.D, dtype=int)
        self.am = torch.nn.parameter.Parameter(
            torch.zeros(2, self.D, dtype=int), requires_grad=False
        )
        self.name = "rp"
        self.input_size = input_size
        self.D = D
        self.num_classes = num_classes
        self.sim_metric = sim_metric
        self.binarize_am = binarize_am
        self.bipolarize_am = bipolarize_am
        self.binarize_hv = binarize_hv
        self.bipolarize_hv = bipolarize_hv

    def encode(self, x, return_time=False):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )

        starter.record()
        hv = self.rp_layer(x.float())
        # hv = torch.where(hv > 0, 1.0, -1.0).to(torch.int32)

        if self.bipolarize_hv:
            hv = bipolarize(hv)

        if self.binarize_hv:
            hv = binarize(hv)
        ender.record()
        torch.cuda.synchronize()

        # convert milliseconds to seconds
        total_time = starter.elapsed_time(ender) / 1000

        if return_time:
            return hv, total_time
        else:
            return hv

    def forward(self, x):
        hv = self.encode(x)
        return hv


class ComboEncoder(HDModel):
    def __init__(
        self,
        input_size: int,
        D: int,
        num_classes: int,
        sim_metric: str,
        binarize_am=False,
        bipolarize_am=False,
        binarize_hv=False,
        bipolarize_hv=False,
    ):
        super(ComboEncoder, self).__init__(
            D=D,
            sim_metric=sim_metric,
            binarize_am=binarize_am,
            bipolarize_am=bipolarize_am,
            binarize_hv=binarize_hv,
            bipolarize_hv=bipolarize_hv,
        )
        self.rp_layer = nn.Linear(input_size, D, bias=False).float()
        init_rp_mat = (
            torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float() * 2 - 1
        )
        self.rp_layer.weight = nn.parameter.Parameter(
            init_rp_mat, requires_grad=False
        ).float()

        self.init_class_hvs = torch.zeros(num_classes, D).float()

        # self.am = torch.zeros(2, self.D, dtype=int)
        self.am = torch.nn.parameter.Parameter(
            torch.zeros(2, self.D, dtype=int), requires_grad=False
        ).to(torch.int32)
        self.name = "combo"

        self.input_size = input_size
        self.D = D
        self.num_classes = num_classes
        self.sim_metric = sim_metric
        self.binarize_am = binarize_am
        self.bipolarize_am = bipolarize_am
        self.binarize_hv = binarize_hv
        self.bipolarize_hv = bipolarize_hv

    def encode(self, x, return_time=False):
        ecfp_hv = x[:, self.input_size :]
        embed = x[:, : self.input_size]

        (
            start,
            end,
        ) = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)
        start.record()

        # would like to avoid capturing this overhead...
        if self.binarize_hv:
            ecfp_hv = binarize(hv)
        if self.bipolarize_hv:
            ecfp_hv = bipolarize(hv)

        embed_hv = self.rp_layer(embed)

        if self.binarize_hv:
            embed_hv = binarize(hv)
        if self.bipolarize_hv:
            embed_hv = bipolarize(hv)

        hv = (ecfp_hv * embed_hv).int()  # bind the hv's together
        hv = torch.where(hv > 0, 1.0, -1.0).to(torch.int32)
        # replace with hv = bipolarize(hv)
        end.record()
        torch.cuda.synchronize()

        t = start.elapsed_time(end) / 1000  # convert to seconds

        if return_time:
            return hv, t
        else:
            return hv

    def forward(self, x):
        hv = self.encode(x)
        return hv


class TokenEncoder(HDModel):
    def __init__(
        self,
        D: int,
        num_classes: int,
        item_mem: dict,
        sim_metric: str,
        binarize_am=False,
        bipolarize_am=False,
        binarize_hv=False,
        bipolarize_hv=True,
    ):
        super(TokenEncoder, self).__init__(
            D=D,
            sim_metric=sim_metric,
            binarize_am=binarize_am,
            bipolarize_am=bipolarize_am,
            binarize_hv=binarize_hv,
            bipolarize_hv=bipolarize_hv,
        )

        self.D = D
        self.num_classes = num_classes
        self.item_mem = item_mem
        self.am = torch.zeros(2, self.D, dtype=int)
        self.name = "molehd"

        self.binarize_am = binarize_am
        self.bipolarize_am = bipolarize_am
        self.binarize_hv = binarize_hv
        self.bipolarize_hv = bipolarize_hv

    def encode(self, tokens: list):
        # tokens is a list of tokens that we will map to item_mem token hvs and produce the smiles hv
        hv = torch.zeros(1, self.D).int()

        batch_tokens = [
            torch.roll(self.item_mem[token], idx).reshape(1, -1)
            for idx, token in enumerate(tokens)
        ]

        hv = torch.vstack(batch_tokens).sum(dim=0).reshape(1, -1)

        # binarize
        # hv = torch.where(hv > 0, hv, -1).int()
        # hv = torch.where(hv <= 0, hv, 1).int()
        # hv = binarize(hv)
        if self.binarize_hv:
            hv = binarize(hv)
        if self.bipolarize_hv:
            hv = bipolarize(hv)

        return hv.to(torch.int32)

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
        return self.RP_encoding(x).to(torch.int32)

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
        self.name = "mlp"

    def forward(self, x, return_time=False):
        if return_time:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()
            out = self.fc_layers(x)
            out = torch.nn.LogSoftmax(dim=1)(out)
            ender.record()

            torch.cuda.synchronize()
            total_time = (
                starter.elapsed_time(ender) / 1000
            )  # convert milliseconds to seconds

            return out, total_time

        else:
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


# utility functions for training and testing
def train_hdc(model, train_dataloader, device, num_epochs, encode=True):
    am_time_cpu_sum = 0
    am_time_cuda_sum = 0
    retrain_time_cpu_sum = 0
    retrain_time_cuda_sum = 0

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

            cuda_starter, cuda_ender = torch.cuda.Event(
                enable_timing=True
            ), torch.cuda.Event(enable_timing=True)

            am_time_cpu_start = time.perf_counter()
            cuda_starter.record()

            #############################Begin Critical Section############################
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
                    if encode:
                        model.am[class_idx] += (
                            # model.encode(x[class_mask, :]).reshape(-1, model.D).sum(dim=0).int()
                            model.encode(x[class_mask, :])
                            .reshape(-1, model.D)
                            .sum(dim=0)
                        )
                    else:
                        model.am[class_idx] += (
                            (x[class_mask, :]).sum(dim=0).reshape(-1).int()
                        )

            #############################End Critical Section############################

            cuda_ender.record()

            torch.cuda.synchronize()
            # include the GPU time so that we can subtract and take the difference to get CPU execution time
            am_time_cpu_end = time.perf_counter()

            am_time_cpu_sum += am_time_cpu_end - am_time_cpu_start
            am_time_cuda_sum += (
                cuda_starter.elapsed_time(cuda_ender) / 1000
            )  # convert elapsed time in milliseconds to seconds

        retrain_time_cpu_sum = 0
        retrain_time_cuda_sum = 0

        learning_curve = []
        # train_encode_time_list = []

        # HDC-Retrain
        for epoch in range(num_epochs):
            mistake_ct = 0
            # TODO: initialize the associative memory with single pass training instead of the random initialization?

            # epoch_encode_time_total = 0
            for batch in tqdm(train_dataloader, desc=f"training HDC epoch {epoch}"):
                # encode_cuda_starter, encode_cuda_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                x, y, hv = None, None, None

                if model.name == "molehd":
                    x = [x[0] for x in batch]
                    y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                    y = y.squeeze().to(device)

                    encode_time_start = time.perf_counter()
                    encode_cuda_starter.record()

                    if encode:
                        hv = torch.cat([model.encode(z) for z in x])
                    else:
                        hv = x

                    # encode_time_end = time.perf_counter()
                    # epoch_encode_time_total += encode_time_end - encode_time_start

                else:
                    x, y = batch
                    x, y = x.to(device), y.squeeze().to(device)

                    # encode_time_start = time.perf_counter()
                    if encode:
                        hv = model.encode(x)
                    else:
                        hv = x
                    hv = hv.float()
                    # encode_time_end = time.perf_counter()
                    # epoch_encode_time_total += encode_time_end - encode_time_start

                # import pdb
                # pdb.set_trace()
                # tqdm.write(str(hv))
                retrain_cuda_starter, retrain_cuda_ender = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)

                # start the cpu counter outside of the cuda record time so it can capture both
                retrain_time_cpu_start = time.perf_counter()
                retrain_cuda_starter.record()
                # if model.bipolarize_hv:
                #   hv = bipolarize(hv)
                # if model.binarize_hv:
                # hv = binarize(hv)
                if model.binarize_hv:
                    hv = binarize(hv)
                if model.bipolarize_hv:
                    hv = bipolarize(hv)

                y_ = model.predict(hv)  # cosine similarity is done in floating point
                update_mask = torch.abs(y - y_).bool()
                mistake_ct += sum(update_mask)

                if update_mask.shape[0] == 1 and update_mask == False:
                    continue
                elif update_mask.shape[0] == 1 and update_mask == True:
                    model.am[int(update_mask)] += hv.reshape(-1)
                    model.am[int(~update_mask.bool())] -= hv.reshape(-1)
                else:
                    for mistake_hv, mistake_label in zip(
                        hv[update_mask], y[update_mask]
                    ):
                        model.am[int(mistake_label)] += mistake_hv
                        model.am[int(~mistake_label.bool())] -= mistake_hv

                retrain_cuda_ender.record()
                torch.cuda.synchronize()

                # include the GPU time so that we can subtract and take the difference to get CPU execution time
                retrain_time_cpu_end = time.perf_counter()

                retrain_time_cuda_sum += (
                    retrain_cuda_starter.elapsed_time(retrain_cuda_ender) / 1000
                )  # convert elapsed time in milliesecons to seconds
                retrain_time_cpu_sum += (
                    retrain_time_cpu_end - retrain_time_cpu_start
                ) - retrain_time_cuda_sum  # subtract the cuda time from the cpu time to get more accuarate measurement

            learning_curve.append(mistake_ct.cpu().numpy())
            # train_encode_time_list.append(epoch_encode_time_total)

        return {
            "model": model,
            "learning_curve": learning_curve,
            # single_pass_train_time,
            "am_time_cpu_sum": am_time_cpu_sum,
            "am_time_cpu_norm": am_time_cpu_sum / len(train_dataloader.dataset),
            "retrain_time_cpu_sum": retrain_time_cpu_sum,
            "retrain_time_cpu_norm": retrain_time_cpu_sum
            / len(train_dataloader.dataset),
            "am_time_cuda_sum": am_time_cuda_sum,
            "am_time_cuda_norm": am_time_cuda_sum / len(train_dataloader.dataset),
            "retrain_time_cuda_sum": retrain_time_cuda_sum,
            "retrain_time_cuda_norm": retrain_time_cuda_sum
            / len(train_dataloader.dataset)
            # "encode_time_cpu_sum": encode_time_cpu_sum,
            # "encode_time_cpu_norm": encode_time_cpu_norm,
            # "encode_time_cuda_sum": encode_time_cuda_sum,
            # "encode_time_cuda_norm": encode_time_cuda_norm,
            # np.sum(train_encode_time_list) / num_epochs
            # None,
        }


def test_hdc(model, test_dataloader, device, encode=True, return_time_list=False):
    with torch.no_grad():
        model = model.to(device)
        # test_time_list = []

        encode_time_cpu_sum = 0
        encode_time_cuda_sum = 0
        test_time_cpu_sum = 0
        test_time_cuda_sum = 0

        # conf_time_list = []
        target_list = []
        pred_list = []
        conf_list = []

        for batch in tqdm(test_dataloader, desc="testing.."):
            x, y, y_, hv = (
                None,
                None,
                None,
                None,
            )

            # encode_cuda_starter, encode_cuda_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # encode data if necessary
            if model.name == "molehd":
                x = [x[0] for x in batch]

                y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                y = y.squeeze()

                if encode:
                    # batch_encode_cpu_start = time.perf_counter()
                    hv = torch.cat([model.encode(z, return_time=False) for z in x])
                    # batch_

                else:
                    hv = x

            else:
                x, y = batch
                x, y = x.to(device), y.squeeze().to(device)

                if encode:
                    hv = model.encode(x)
                else:
                    hv = x

            y_, batch_cpu_time, batch_cuda_time = model.predict(
                hv.float(), return_time=True
            )

            # TODO: pre-allocate the memory instead of appending to lsits
            target_list.append(y.cpu().reshape(-1, 1))
            pred_list.append(y_.cpu().reshape(-1, 1))

            # test_time_list.append(batch_forward_time)
            test_time_cpu_sum += batch_cpu_time
            test_time_cuda_sum += batch_cuda_time

            conf, batch_conf_time = model.compute_confidence(hv, return_time=True)
            conf_list.append(conf.cpu())
            # conf_time_list.append(batch_conf_time)

        return {
            "y_pred": torch.cat(pred_list),
            "y_true": torch.cat(target_list),
            "eta": torch.cat(conf_list),
            # "test_time_list": np.array(test_time_list)
            "test_time_cpu_sum": test_time_cpu_sum,
            "test_time_cpu_norm": test_time_cpu_sum / len(test_dataloader.dataset),
            "test_time_cuda_sum": test_time_cuda_sum,
            "test_time_cuda_norm": test_time_cuda_sum / len(test_dataloader.dataset),
        }


def encode_hdc(model, dataloader, device, use_numpy=False):
    with torch.no_grad():
        model = model.to(device)
        # model.am = model.am.float()
        encode_list = []

        encode_time_list = []
        target_list = []

        for batch in tqdm(dataloader, desc="encoding.."):
            x, y, hv = (
                None,
                None,
                None,
            )
            batch_encode_time = None

            if model.name == "molehd":
                x = [x[0] for x in batch]

                y = torch.from_numpy(np.array([x[1] for x in batch])).int()
                y = y.squeeze()

                hv_list = [model.encode(z, return_time=True) for z in x]
                hv = torch.cat([h[0] for h in hv_list])
                batch_encode_time = torch.sum([h[1] for h in hv_list]).item()

            else:
                x, y = batch
                x = x.to(device)

                hv, batch_encode_time = model.encode(x, return_time=True)

            if use_numpy:
                encode_list.append(hv)
                target_list.append(y)
            else:
                encode_list.append(hv.cpu().numpy())
                target_list.append(y.cpu().numpy())

            encode_time_list.append(batch_encode_time)

    encode_list = [x.cpu() for x in encode_list]
    return (
        torch.cat(encode_list).to(torch.int32),
        torch.cat(target_list).to(torch.int32),
        np.array(encode_time_list),
    )


def run_hdc(
    model,
    config,
    epochs,
    batch_size,
    num_workers,
    n_trials,
    random_state,
    train_dataset,
    test_dataset,
    smiles_train=None,
    smiles_test=None,
    encode=True,
    result_dict=None,  # can use this to finish computing a partial result
    result_path=None,
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
        train_encode_start = time.perf_counter()
        train_dataset_hvs = model.encode_dataset(train_toks)
        train_encode_time = time.perf_counter() - train_encode_start

        test_encode_start = time.perf_counter()
        test_dataset_hvs = model.encode_dataset(test_toks)
        test_encode_time = time.perf_counter() - test_encode_start

        train_dataset_hvs = torch.vstack(train_dataset_hvs).int()
        test_dataset_hvs = torch.vstack(test_dataset_hvs).int()

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
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=False,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if result_dict is None:
        result_dict = {"trials": {}}

    # print(f"result_dict: {result_dict}")
    # avoid recomputing trials that have finished
    comp_trial_ct = len(result_dict["trials"])
    print(f"strarting to train from trial: {comp_trial_ct}")
    for i in range(comp_trial_ct, n_trials):
        trial_dict = {}

        # this should force each call of .fit to be different...I think..
        seed_rngs(random_state + i)

        # model, learning_curve, single_pass_train_time, retrain_time, _ = train_hdc(
        # model=model,
        # train_dataloader=train_dataloader,
        # num_epochs=epochs,
        # device=config.device,
        # encode=encode,
        # )

        train_dict = train_hdc(
            model=model,
            train_dataloader=train_dataloader,
            num_epochs=epochs,
            device=config.device,
            encode=encode,
        )

        trial_dict["hd_learning_curve"] = train_dict["learning_curve"]

        # time test inside of the funcion
        test_dict = test_hdc(
            model=train_dict["model"],
            test_dataloader=test_dataloader,
            device=config.device,
            encode=encode,
        )

        trial_dict["am"] = {
            0: train_dict["model"].am[0].cpu().numpy(),
            1: train_dict["model"].am[1].cpu().numpy(),
        }  # store the associative memory so it can be loaded up later on

        trial_dict["y_pred"] = test_dict["y_pred"].cpu().numpy()
        trial_dict["eta"] = test_dict["eta"].cpu().numpy().reshape(-1, 1)
        trial_dict["y_true"] = test_dict["y_true"].cpu().numpy()

        # trial_dict["single_pass_train_time"] = single_pass_train_time
        trial_dict["am_time_cpu_sum"] = train_dict["am_time_cpu_sum"]
        trial_dict["am_time_cpu_norm"] = train_dict["am_time_cpu_norm"]
        trial_dict["retrain_time_cpu_sum"] = train_dict["retrain_time_cpu_sum"]
        trial_dict["retrain_time_cpu_norm"] = train_dict["retrain_time_cpu_norm"]

        trial_dict["am_time_cuda_sum"] = train_dict["am_time_cuda_sum"]
        trial_dict["am_time_cuda_norm"] = train_dict["am_time_cuda_norm"]
        trial_dict["retrain_time_cuda_sum"] = train_dict["retrain_time_cuda_sum"]
        trial_dict["retrain_time_cuda_norm"] = train_dict["retrain_time_cuda_norm"]

        trial_dict["train_time_cpu_sum"] = (
            trial_dict["am_time_cpu_sum"] + trial_dict["retrain_time_cpu_sum"]
        )
        trial_dict["train_time_cuda_sum"] = (
            trial_dict["am_time_cuda_sum"] + trial_dict["retrain_time_cuda_sum"]
        )

        trial_dict["train_time_cpu_norm"] = (
            trial_dict["am_time_cpu_norm"] + trial_dict["retrain_time_cpu_norm"]
        )
        trial_dict["train_time_cuda_norm"] = (
            trial_dict["am_time_cuda_norm"] + trial_dict["retrain_time_cuda_norm"]
        )

        # trial_dict["test_time"] = test_dict["test_time_list"]
        trial_dict["test_time_cpu_sum"] = test_dict["test_time_cpu_sum"]
        trial_dict["test_time_cpu_norm"] = test_dict["test_time_cpu_norm"]
        trial_dict["test_time_cuda_sum"] = test_dict["test_time_cuda_sum"]
        trial_dict["test_time_cuda_norm"] = test_dict["test_time_cuda_norm"]
        # trial_dict["conf_test_time"] = test_dict["conf_test_time"]
        # trial_dict["train_encode_time"] = test_encode_time
        # trial_dict["test_encode_time"] = test_encode_time
        # trial_dict["encode_time"] = train_encode_time + test_encode_time

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
        # import pdb
        # pdb.set_trace()

        trial_dict["enrich-1"] = compute_enrichment_factor(
            scores=trial_dict["eta"], labels=trial_dict["y_true"], n_percent=0.01
        )
        trial_dict["enrich-10"] = compute_enrichment_factor(
            scores=trial_dict["eta"], labels=trial_dict["y_true"], n_percent=0.10
        )

        print(trial_dict["class_report"])
        print(f"roc-auc {trial_dict['roc-auc']}")

        validate(
            labels=trial_dict["y_true"],
            pred_labels=trial_dict["y_pred"],
            pred_scores=trial_dict["eta"],
        )

        # store the new result then save it
        result_dict["trials"][i] = trial_dict
        print(f"saving trial {i} to {result_path}")
        torch.save(result_dict, result_path)

    return result_dict


def train_mlp(model, train_dataloader, epochs, device):
    model = model.to(device)

    forward_cpu_time = 0.0
    forward_cuda_time = 0.0
    loss_cpu_time = 0.0
    loss_cuda_time = 0.0
    backward_cpu_time = 0.0
    backward_cuda_time = 0.0

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc=f"training MLP epoch: {epoch}"):
            model.optimizer.zero_grad()

            x, y = batch

            x = x.to(device).float()
            y = y.to(device).reshape(-1).long()

            forward_cuda_starter, forward_cuda_ender = torch.cuda.Event(
                enable_timing=True
            ), torch.cuda.Event(enable_timing=True)
            forward_cpu_start = time.perf_counter()
            forward_cuda_starter.record()
            y_ = model(x)
            forward_cuda_ender.record()
            torch.synchronize()
            forward_cpu_end = time.perf_counter()

            forward_cpu_time += forward_cpu_end - forward_cpu_start
            forward_cuda_time += (
                forward_cuda_starter.elapsed_time(forward_cuda_ender) / 1000
            )  # convert elapsed time in milliseconds to seconds

            # check that the output is correct dimension
            y_ = y_.reshape(-1, 2)

            loss_cuda_starter, loss_cuda_ender = torch.cuda.Event(
                enable_timing=True
            ), torch.cuda.Event(enable_timing=True)
            loss_cpu_start = time.perf_counter()
            loss_cuda_starter.record()
            # loss_start = time.perf_counter()
            loss = model.criterion(y_, y)
            loss_cuda_ender.record()
            torch.cuda.synchronize()
            # loss_end = time.perf_counter()
            loss_cpu_end = time.perf_counter()

            loss_cpu_time += loss_cpu_end - loss_cpu_start
            loss_cuda_time += loss_cuda_starter.elapsed_time(loss_cuda_ender) / 1000

            backward_cuda_starter, backward_cuda_ender = torch
            # backward_start = time.perf_counter()
            loss.backward()
            # backward_end = time.perf_counter()

            model.optimizer.step()

            # forward_time += forward_end - forward_start
            # loss_time += loss_end - loss_start
            # backward_time += backward_end - backward_start

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
    test_time_list = []
    preds = []
    targets = []

    # warm up GPU

    batch_size = val_dataloader.batch_size
    if batch_size >= len(val_dataloader.dataset):
        print(
            "batch size is larger than input dataset, changing this to input dataset size"
        )
        batch_size = len(val_dataloader.dataset)

    dummy_input = torch.zeros(
        val_dataloader.batch_size, val_dataloader.dataset[0][0].shape[0], device=device
    )

    for _ in tqdm(range(10), desc="warming up GPU"):
        model.forward(dummy_input)

    batch_ct = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"validating MLP"):
            x, y = batch

            targets.append(y.cpu())

            x = x.to(device).float()
            y = y.to(device).reshape(-1).long()

            y_, batch_forward_time = model.forward(x, return_time=True)

            preds.append(y_.cpu())
            loss_start = time.perf_counter()
            loss = model.criterion(y_, y)
            loss_end = time.perf_counter()

            total_loss += loss.item()

            forward_time += batch_forward_time
            loss_time += loss_end - loss_start

            test_time_list.append(batch_forward_time)

            batch_ct += 1
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    N = len(val_dataloader.dataset)

    return {
        "y_true": targets.to("cpu"),
        "y_pred": torch.argmax(preds, dim=1).to("cpu"),
        "eta": preds.reshape(-1, 2).to("cpu"),
        "loss": total_loss,
        "forward_time": forward_time,
        "loss_time": loss_time,
        "test_time_list": np.array(test_time_list),
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


def run_mlp(
    config,
    batch_size,
    epochs,
    num_workers,
    n_trials,
    random_state,
    train_dataset,
    test_dataset,
):
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
        # "lr": tune.choice([1e-3, 1e-2],
        "lr": tune.uniform(1e-5, 1e-1),
        "activation": tune.choice([torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.GELU()]),
        "criterion": tune.choice([torch.nn.NLLLoss()]),
        "optimizer": tune.choice([torch.optim.Adam, torch.optim.SGD]),
    }

    from torch.utils.data import DataLoader, SubsetRandomSampler
    from sklearn.model_selection import StratifiedShuffleSplit

    # Assume you have a PyTorch dataset named 'dataset' and corresponding labels 'labels'

    # Define the number of splits and the train/validation split ratio
    n_splits = 1  # You can change this according to your requirement
    test_size = 0.2  # Ratio of validation data

    # Initialize Stratified Shuffle Split
    stratified_splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42
    )

    # Get indices for train and validation sets
    train_indices, val_indices = next(
        stratified_splitter.split(
            np.zeros(len(train_dataset.tensors[1])), train_dataset.tensors[1]
        )
    )

    # Define samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        # shuffle=True,#mutually exclusive with SubsetRandomSampler
        sampler=train_sampler,
    )

    val_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        persistent_workers=False,
        # shuffle=False, #mutually exclusive with SubsetRandomSampler
        sampler=val_sampler,
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
            # train_dataloader=train_dataloader,
            train_dataloader=train_dataloader,
            epochs=epochs,
            device=config.device,
        )
        test_dict = val_mlp(
            model=train_dict["model"],
            val_dataloader=test_dataloader,
            device=config.device,
        )

        seed = random_state + i
        # this should force each call of .fit to be different...I think..
        seed_rngs(seed)
        # collect the best parameters and train on full training set, we capture timings wrt to the optimal configuration
        # construct after seeding the rng

        trial_dict["y_pred"] = test_dict["y_pred"].cpu().numpy()
        trial_dict["eta"] = test_dict["eta"].cpu().numpy()
        trial_dict["y_true"] = test_dict[
            "y_true"
        ].numpy()  # these were being saved as torch arrays, may slow down notebooks
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

        try:
            trial_dict["roc-auc"] = roc_auc_score(
                y_score=trial_dict["eta"][:, 1], y_true=trial_dict["y_true"]
            )

        except ValueError as e:
            trial_dict["roc-auc"] = None
            print(e)
        # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

        trial_dict["enrich-1"] = compute_enrichment_factor(
            scores=trial_dict["eta"][:, 1], labels=trial_dict["y_true"], n_percent=0.01
        )
        trial_dict["enrich-10"] = compute_enrichment_factor(
            scores=trial_dict["eta"][:, 1], labels=trial_dict["y_true"], n_percent=0.10
        )

        print(trial_dict["class_report"])
        print(f"roc-auc {trial_dict['roc-auc']}")

        # enrichment metrics

        validate(
            labels=trial_dict["y_true"],
            pred_labels=trial_dict["y_pred"],
            pred_scores=trial_dict["eta"][:, 1],
        )
        result_dict["trials"][i] = trial_dict

    return result_dict


def get_model(config):
    assert not (config.bipolarize_am and config.binarize_am)
    assert not (config.bipolarize_hv and config.binarize_hv)

    if config.model == "molehd":
        model = TokenEncoder(
            D=config.D,
            num_classes=2,
            sim_metric=config.sim_metric,
            binarize=config.binarize,
            bipolarize=config.bipolarize,
        )
        # will update item_mem after processing input data

    elif config.model == "selfies":
        from hdpy.selfies_enc.encode import SELFIESHDEncoder

        model = SELFIESHDEncoder(
            D=config.D,
            sim_metric=config.sim_metric,
            binarize_am=config.binarize_am,
            bipolarize_am=config.bipolarize_am,
            binarize_hv=config.binarize_hv,
            bipolarize_hv=config.bipolarize_hv,
        )

    elif config.model == "ecfp":
        from hdpy.ecfp.encode import StreamingECFPEncoder

        model = StreamingECFPEncoder(
            D=config.D,
            radius=config.ecfp_radius,
            sim_metric=config.sim_metric,
            binarize_am=config.binarize_am,
            bipolarize_am=config.bipolarize_am,
            binarize_hv=config.binarize_hv,
            bipolarize_hv=config.bipolarize_hv,
        )

    elif config.model == "rp":
        # assert config.ecfp_length is not None
        assert config.D is not None
        model = RPEncoder(
            input_size=config.input_size,
            D=config.D,
            num_classes=2,
            sim_metric=config.sim_metric,
            binarize_am=config.binarize_am,
            bipolarize_am=config.bipolarize_am,
            binarize_hv=config.binarize_hv,
            bipolarize_hv=config.bipolarize_hv,
        )

    elif config.model == "mlp-small":
        model = MLPClassifier(
            layer_sizes=((config.ecfp_length, 128), (128, 2)),
            lr=1e-3,
            activation=torch.nn.ReLU(),
            criterion=torch.nn.NLLLoss(),
            optimizer=torch.optim.Adam,
        )

    elif config.model == "mlp-large":
        model = MLPClassifier(
            layer_sizes=((config.ecfp_length, 512), (512, 256), (256, 128), (128, 2)),
            lr=1e-3,
            activation=torch.nn.ReLU(),
            criterion=torch.nn.NLLLoss(),
            optimizer=torch.optim.Adam,
        )
    elif config.model == "directecfp":
        model = HDModel(
            D=config.D,
            name="directecfp",
            sim_metric=config.sim_metric,
            binarize_am=config.binarize_am,
            bipolarize_am=config.bipolarize_am,
            binarize_hv=config.binarize_hv,
            bipolarize_hv=config.bipolarize_hv,
        )
        model.am = torch.zeros(2, model.D, dtype=float)

        if config.binarize_am:
            model.am = binarize(model.am)

        if config.bipolarize_am:
            model.am = bipolarize(model.am)

    elif config.model == "combo":
        model = ComboEncoder(
            input_size=config.input_size,
            D=config.D,
            num_classes=2,
            sim_metric=config.sim_metric,
            binarize_am=config.binarize_am,
            bipolarize_am=config.bipolarize_am,
            binarize_hv=config.binarize_hv,
            bipolarize_hv=config.bipolarize_hv,
        )
    else:
        # if using sklearn or pytorch non-hd model
        model = None

    return model

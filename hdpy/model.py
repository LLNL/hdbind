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
from torch.utils.data import Dataset
# from hdpy.utils import CustomDataset
import numpy as np


class HDModel(nn.Module):

    def __init__(self, D:int):
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


    # def encode_dataset(self, dataset):
        # lets process batches instead of one by one
        # dataset_hvs = []

        # for datapoint in tqdm(dataset, desc="encoding dataset..."):
            # dataset_hvs.append(self.encode(datapoint).reshape(1, -1)) 

    
        # torch.cat(dataset_hvs)


        # return dataset_hvs


    def predict(self, enc_hvs):
        # import ipdb
        # ipdb.set_trace()
        # preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().float(), torch.cat([x.reshape(1,-1) for x in self.am.values()]).float()), dim=1)
        preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().cuda().float(), self.am.clone().cuda().float()), dim=1)
        return preds

    def forward(self, x):

        out = self.predict(x) 
        return out


    def compute_confidence(self, dataset_hvs):

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        # am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        # this torchmetrics function potentially edits in place so we make a clone
        sims = torchmetrics.functional.pairwise_cosine_similarity(dataset_hvs.clone().cuda().float(), self.am.clone().cuda().float())

        eta = (sims[:, 1] - sims[:, 0]) * (1/4)
        eta = torch.add(eta, (1/2))
        return eta.reshape(-1)

    def retrain(self, dataset_hvs, labels, return_mistake_count=False, lr=1.0):

        # should do this in parallel instead of the sequential? or can we define two functions and possibly combine the two? i.e. use the parallel version most of the time and then periodically update with the sequential version?

        shuffle_idx = torch.randperm(dataset_hvs.size()[0])
        dataset_hvs = dataset_hvs[shuffle_idx].int()
        labels = labels[shuffle_idx].int()

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        # am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        mistakes = 0

        # import pdb
        # pdb.set_trace() 
        for hv, label in tqdm(zip(dataset_hvs, labels), total=len(dataset_hvs)):
        
            out = int(torch.argmax(torch.nn.CosineSimilarity()(hv.float(),self.am.float())))

            if out == int(label):
                pass
            else:
                mistakes += 1
                self.am[out] -= (lr*hv).int()
                self.am[int(label)] += (lr*hv).int()

        if return_mistake_count:
            return mistakes


    def fit(self, x_train, y_train, num_epochs, lr=1.0):
        for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
            self.train_step(train_features=x_train, train_labels=y_train, lr=lr)

'''
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
'''    

class RPEncoder(HDModel):
    def __init__(self, input_size:int, D:int, num_classes:int):
        super(RPEncoder, self).__init__(D=D)
        # super()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float()*2-1
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float()

        self.am = torch.zeros(2,self.D, dtype=int)



    def encode(self, x):
                
        hv = self.rp_layer(x.float())
        hv = torch.where(hv>0, 1.0, -1.0).int()
        return hv 




class TokenEncoder(HDModel):

    def __init__(self, D:int, num_classes:int, item_mem=dict):

        super(TokenEncoder, self).__init__(D=D)

        self.D = D
        self.num_classes = num_classes
        self.item_mem = item_mem
        self.am = torch.zeros(2,self.D, dtype=int)
    
    def encode(self, tokens:list):

        # tokens is a list of tokens that we will map to item_mem token hvs and produce the smiles hv
        # import pdb
        # pdb.set_trace()
        hv = torch.zeros(1, self.D).int()

        #todo: try using a list comprehension instead

        batch_tokens = [torch.roll(self.item_mem[token], idx).reshape(1,-1) for idx, token in enumerate(tokens)]
        # for idx, token_hv in enumerate(batch_tokens):
            # token_hv = self.item_mem[token]
            # hv = hv + torch.roll(token_hv, idx).int()
        hv = torch.vstack(batch_tokens).sum(dim=0).reshape(1,-1)

        # binarize
        hv = torch.where(hv > 0, hv, -1).int()
        hv = torch.where(hv <= 0, hv, 1).int()

        return hv


    def forward(self, x):
        return super().forward(x).cpu()

    def encode_batch(self, token_list:list):

        return torch.cat([self.encode(z) for z in token_list])










class ECFPDataset(Dataset):
    def __init__(self, input_size:int, radius:float, D:int, num_classes:int, fp_list:list):
        super()

        # self.smiles_list = smiles_list
        # self.rp_encoder = RPEncoder(input_size=input_size, num_classes=num_classes, D=D)
        # self.rp_encoder = rp_encoder
        self.input_size = input_size
        self.radius = radius
        self.D = D
        self.num_classes = num_classes
        # self.smiles_list = smiles_list
        # self.compute_fingerprint_from_smiles = compute_fingerprint_from_smiles
        # import ipdb
        # ipdb.set_trace()        
        self.ecfp_arr = torch.from_numpy(np.concatenate(fp_list)).reshape(-1, self.input_size)
        

    def __len__(self):
        return len(self.ecfp_arr)

    def __getitem__(self, idx):
        # import ipdb 
        # ipdb.set_trace()
        # ecfp = torch.from_numpy(self.compute_fingerprint_from_smiles(smiles=self.smiles_list[idx],
                                                    # radius=self.radius,
                                                    # input_size=self.input_size))
        # ecfp = self.ecfp_list[idx]
        return self.ecfp_arr[idx]
        # if ecfp is not None:
            # print(ecfp)
            # start = time.time()
            # hv = self.rp_encoder.encode(ecfp)
            # end = time.time()

            # return hv, end-start
        # else:
            # self.smiles_list.pop(idx)
            # return None



class HD_Sparse_Classification(HDModel):
    def __init__(self, input_size, D, density, num_classes):
        super(HD_Sparse_Classification, self).__init__()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        init_rp_mat = torch.bernoulli(torch.tensor([[density] * input_size] * D)).float()
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)
        
        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()
        # self.init_class_hvs = torch.sparse_coo_tensor(num_classes, D, (num_class)).cuda()



    def RP_encoding(self, x):
        out = self.rp_layer(x)
        # out = torch.where(out>0, 1.0, -1.0)
        return out

    def encode(self, x):
        return self.RP_encoding(x)

    def init_class(self, x_train, train_labels):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[train_labels[i]] += out[i]
        # self.am = nn.parameter.Parameter(self.init_class_hvs, requires_grad=True)
        self.am = self.init_class_hvs 


class HD_Kron_Classification(HDModel):
    def __init__(self, Kron_shape, input_size, D, num_classes, binary=True):
        super(HD_Kron_Classification, self).__init__()
        self.Kron_shape = Kron_shape
        self.D, self.F = D, input_size
        self.binary = binary

        self.Kron_1 = nn.Linear(Kron_shape[1], Kron_shape[0] , bias=False)
        self.Kron_2 = nn.Linear(self.F//Kron_shape[1], self.D//Kron_shape[0], bias=False)
        self.init_class_hvs = torch.zeros(num_classes, self.D).float().cuda()

        if binary:
            init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * Kron_shape[1]] * Kron_shape[0])).float()*2-1
            self.Kron_1.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)
            init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * (self.F//Kron_shape[1])] * (self.D//Kron_shape[0]))).float()*2-1
            self.Kron_2.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

    def RP_encoding(self, x):

        x = x.view(x.size()[0], self.F//self.Kron_shape[1], self.Kron_shape[1])
        out = self.Kron_1(x)
        out = self.Kron_2(out.permute(0, 2, 1))
        out = out.view(out.size()[0], -1)
        
        if self.binary:
            out = torch.where(out>0, 1.0, -1.0)
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
        self.quantize_scale = 1./quan_level

        density = 0.5

        init_rp_mat = torch.bernoulli(torch.tensor([[density] * input_size] * D)).float()
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)
        
        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()

    def quantize(self, x):
        return torch.fake_quantize_per_tensor_affine(x, scale=self.quantize_scale, zero_point=0, quant_min=0, quant_max=3)

    #def encoding(self, x):
    #    out = self.rp_layer(x)
        # out = torch.where(out>0, 1.0, -1.0)
    #    return out

    def encode(self, x):
        return self.RP_encoding(x)

    def RP_encoding(self, x):

        # ipdb.set_trace()
        out = self.rp_layer(x)
        out = torch.where(out>0, 1.0, -1.0)
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
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, lr):
        super(ClassifierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out).softmax(dim=1)
        return out


    # def fit(self, x_train, y_train, num_epochs):

        # cross entropy likes the long tensor so just do this once before training instead of multiple times
        # y_train = y_train.long()

        # Train the model

        # train_dataloader = DataLoader(CustomDataset(x_train, y_train), batch_size=32)

        # for batch in tqdm(train_dataloader, desc="training MLP..."):
            # Forward pass
            # features, labels = batch
            # self.optimizer.zero_grad()
            # outputs = self.forward(features)
            # loss = self.criterion(outputs, labels)
            # print(loss) 
            # Backward and optimize
            # loss.backward()
            # self.optimizer.step()


    def predict(self, features):
        return self.forward(features)

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
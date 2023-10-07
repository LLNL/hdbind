import torch
import torch.nn as nn
from tqdm import tqdm 
import torchmetrics


def get_random_hv(m, n):
    return torch.bernoulli(torch.tensor([[0.5] * m] * n)).float()*2-1



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
        preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().float(), self.am.clone()), dim=1)
        return preds 

    def forward(self, x):

        out = self.predict(x) 
        return out


    def compute_confidence(self, dataset_hvs):

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        # am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        # this torchmetrics function potentially edits in place so we make a clone
        sims = torchmetrics.functional.pairwise_cosine_similarity(dataset_hvs.clone(), self.am.clone())

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


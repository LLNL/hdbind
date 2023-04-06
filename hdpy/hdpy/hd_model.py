import torch
import torch.nn as nn
from tqdm import tqdm 
import torchmetrics
from torch.utils.data import Dataset


def binarize(x):
    return torch.where(x>0, 1.0, -1.0)


class CustomDataset(Dataset):
    def __init__(self, features, labels):

        self.features = features 
        self.labels = labels 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class HDModel(nn.Module):

    def __init__(self):
        super(HDModel, self).__init__()
        self.am = None

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

    def encode(self, x):
        raise NotImplementedError("Please implement this function in a subclass")


    def encode_dataset(self, dataset):

        dataset_hvs = []

        for datapoint in tqdm(dataset, desc="encoding dataset..."):
            dataset_hvs.append(self.encode(datapoint).reshape(1, -1)) 

    
        torch.cat(dataset_hvs)


        return dataset_hvs


    def predict(self, enc_hvs):

        import pdb 
        pdb.set_trace()
        preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs.clone().float(), torch.cat([x.reshape(1,-1) for x in self.am.values()]).float()), dim=1)

        return preds 

    def forward(self, x):

        out = self.predict(x) 
        return out


    def compute_confidence(self, dataset_hvs):

        # TODO: parallelize the cosine similarity calc

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        sims = torchmetrics.functional.pairwise_cosine_similarity(dataset_hvs.clone(), am_array)

        # eta_list = []

        '''
        for idx, hv in enumerate(dataset_hvs):
            # out = torch.nn.CosineSimilarity()(am_array, hv.float())
            out = sims[idx,:]
            eta = 0
            try:
                eta = (1/2) + (1/4) * (out[1] - out[0])
            except Exception as e:
                print(e)
                eta = 0
            eta_list.append(eta)

        # return torch.cat([x.reshape(1, -1) for x in eta_list]).cuda()
        return torch.cat([x.reshape(1, -1) for x in eta_list])
        '''

        eta = (sims[:, 1] - sims[:, 0]) * (1/4)
        eta = torch.add(eta, (1/2))
        return eta.reshape(-1)

    def retrain(self, dataset_hvs, labels, return_mistake_count=False, lr=1.0):

        shuffle_idx = torch.randperm(dataset_hvs.size()[0])
        dataset_hvs = dataset_hvs[shuffle_idx].int()
        labels = labels[shuffle_idx].int()

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)


        mistakes = 0
        
        for hv, label in tqdm(zip(dataset_hvs, labels), desc=f"retraining...", total=len(dataset_hvs)):

            out = int(torch.argmax(torch.nn.CosineSimilarity()(am_array.float(), hv.float())))

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

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

        
        # import ipdb 
        # ipdb.set_trace()
        torch.cat(dataset_hvs)


        return dataset_hvs


    def predict(self, enc_hvs):
        # import pdb 
        # pdb.set_trace()

        # preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs, self.am.values()), dim=1)
        preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs, torch.cat([x.reshape(1,-1) for x in self.am.values()])), dim=1)

        return preds 

    def forward(self, x):

        out = self.predict(x) 
        return out


    def compute_confidence(self, dataset_hvs):

        # TODO: parallelize the cosine similarity calc

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)


        eta_list = []

        for hv in dataset_hvs:
            out = torch.nn.CosineSimilarity()(am_array, hv)
            eta = 0
            try:
                eta = (1/2) + (1/4) * (out[1] - out[0])
            except Exception as e:
                print(e)
                eta = 0
            eta_list.append(eta)

        return torch.cat([x.reshape(1, -1) for x in eta_list]).cuda()



    def retrain(self, dataset_hvs, labels, lr=1.0):

        shuffle_idx = torch.randperm(dataset_hvs.size()[0])
        dataset_hvs = dataset_hvs[shuffle_idx]
        labels = labels[shuffle_idx]
        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        for hv, label in tqdm(zip(dataset_hvs, labels), desc=f"retraining...", total=len(dataset_hvs)):

            
            out = int(torch.argmax(torch.nn.CosineSimilarity()(am_array, hv)))

            if out == int(label):
                pass
            else:
                self.am[out] -= lr*hv
                self.am[int(label)] += lr*hv



        '''
        shuffle_idx = torch.randperm(train_features.size()[0])
        train_features = train_features[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        enc_hvs = self.encode(train_features)

        preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs, self.am), dim=1)


        #parallelized version below but has some bugs/open areas for improvement
        misclass_mask = preds != train_labels

        # find the misclassified examples and subtract from the associative memory, leverage associative (abuse of words) property of addition/subtraction 
        am_array = torch.index_select(self.am, 0, preds[misclass_mask]) - enc_hvs[misclass_mask]

        # ipdb.set_trace()
        for label in train_labels.int().unique():

            # train_labels[misclass_mask] gives the true value for a misclassified training example, we select the examples corresponding to a
            # particular class, then sum their values in from am_array to produce a 1 by D array/vector, we multiply this by the learning 
            # rate to dampen the values of the update and add the result to the corresponding entry in the associative memory
            self.am[int(label)] += lr * am_array[train_labels[misclass_mask] == label].sum(dim=0)

        self.am = binarize(self.am) 


        '''

    def fit(self, x_train, y_train, num_epochs, lr=1.0):
        for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
            self.train_step(train_features=x_train, train_labels=y_train, lr=lr)

import time
import torch
# import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
# from rdkit import Chem
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import deepchem as dc
from hdpy.ecfp import compute_fingerprint_from_smiles


class timing_part:
    def __init__(self, TAG, verbose=False):
        self.TAG = str(TAG)
        self.total_time = 0
        self.start_time = 0
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        exit_time = time.time()
        self.total_time = exit_time - self.start_time
        if self.verbose:
            tqdm.write(f"{self.TAG}\t{self.total_time}")


def load_molformer_embeddings(embed_path):
    # with open(embed_path, "rb") as handle:
    data = torch.load(embed_path)
    return data['embeds'], data['labels']
    # rep_list = []
    # label_list = []
    # time_list = []

    # for batch_idx, batch in enumerate(data):
        # batch_reps = batch[0]
        # import pdb
        # pdb.set_trace()
 
        # reps = torch.cat([batch_reps[x].mean(dim=1) for x in range(len(batch_reps))], dim=0)
        # rep_list.append(reps)
        # label_list.append(torch.cat([torch.from_numpy(np.asarray(x)) for x in batch[2]]))
                            
        # time = sum(batch[1]) / sum([batch[0][x].shape[0] for x in range(len(batch[0]))])
            
        # time_list.append(time)
            
    # return torch.cat(rep_list, dim=0), torch.cat(label_list, dim=0), time_list
    


def load_features(path:str, dataset:str):

    features, labels = None, None
    data = np.load(path)

    if dataset == "pdbbind":
        # map the experimental -logki/kd value to a discrete category
        features = data[:, :-1]
        labels = data[:, -1]
        labels = np.asarray([convert_pdbbind_affinity_to_class_label(x) for x in labels])


        binary_label_mask = labels != 2

        return features[binary_label_mask], labels[binary_label_mask]

    elif dataset == "dude":

        features = data[:, :-1]
        labels = data[:, -1]
              
        labels = 1- labels
    else:
        raise NotImplementedError("specify a valid supported dataset type")


    # import ipdb 
    # ipdb.set_trace()
    return features, labels


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
    



from torch.utils.data import Dataset


class SMILESDataset(Dataset):

    def __init__(self, split_df):

        pass



class MolFormerDataset(Dataset):

    def __init__(self, path, split_df, smiles_col):
        super()
        self.path = path 
        self.smiles_col = smiles_col
        # self.label_col = label_col
        embeds = torch.load(path)

        x_train, x_test, y_train, y_test = [], [], [], []

        self.train_idxs, self.test_idxs = [], []
        for group, group_df in split_df.groupby("split"):
            split_idxs = group_df["index"].values

            # embed_idxs is the index_values we stored when running the molformer extraction code

            for idx in split_idxs:

                embed_idx_mask = np.equal(idx, embeds["idxs"])
                embed = embeds["embeds"][embed_idx_mask]
                label = embeds["labels"][embed_idx_mask]


                if group == "train":
                    # self.train_idxs = split_idxs
                    x_train.append(embed)
                    y_train.append(label)
                else:
                    # x_test.append(embeds["embeds"][idx])
                    x_test.append(embed)
                    y_test.append(label)

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.concatenate(y_train).reshape(-1,1)
        y_test = np.concatenate(y_test).reshape(-1,1)
        

        self.x = np.vstack([x_train, x_test])
        self.y = np.vstack([y_train, y_test]).astype(int)


        self.train_idxs = np.asarray(list(range(len(x_train))))
        self.test_idxs = np.asarray(list(range(len(x_train), len(x_train)+len(x_test))))


        # import ipdb 
        # ipdb.set_trace()

        self.smiles_train = split_df[split_df["split"] == "train"][self.smiles_col]

        self.smiles_test = split_df[split_df["split"] == "test"][self.smiles_col]
        self.smiles = pd.concat([self.smiles_train, self.smiles_test])

        self.x_train = torch.from_numpy(x_train).int()
        self.x_test = torch.from_numpy(x_test).int()
        self.y_train = torch.from_numpy(y_train).int()
        self.y_test = torch.from_numpy(y_test).int()


    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

    def get_train_test_splits(self):


        # import ipdb
        # ipdb.set_trace()
        return self.x[self.train_idxs], self.x[self.test_idxs], self.y[self.train_idxs], self.y[self.test_idxs]




class ECFPDataset(Dataset):

    def __init__(self, path, smiles_col:str, label_col:str,
                  split_df:pd.DataFrame, split_type:str, ecfp_length:int, 
                  ecfp_radius:int, random_state:int, smiles:np.array, labels:list):
        super()

        # import pdb
        # pdb.set_trace()
        self.path = path
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.random_state = random_state
        self.split_df = split_df
        self.split_type = split_type
        self.ecfp_length = ecfp_length
        self.ecfp_radius = ecfp_radius
        self.labels = labels
        
        
        self.fps = np.asarray([
                compute_fingerprint_from_smiles(x, length=ecfp_length, radius=ecfp_radius)
                for x in tqdm(split_df[self.smiles_col].values.tolist())
            ])

        valid_idxs = np.array([idx for idx, x in enumerate(self.fps) if x is not None])

        self.split_df = split_df.iloc[valid_idxs]

        self.smiles = smiles

        self.smiles_train = self.smiles[self.split_df[self.split_df["split"] == "train"]["index"]]
        self.smiles_test = self.smiles[self.split_df[self.split_df["split"] == "test"]["index"]]

        #todo: do this with a pool 
        self.x_train =  np.concatenate([
                compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                for x in tqdm(self.smiles_train)
            ], axis=0)

        #todo: do this with a pool 
        self.x_test =  np.concatenate([
                compute_fingerprint_from_smiles(x, length=self.ecfp_length, radius=self.ecfp_radius).reshape(1,-1)
                for x in tqdm(self.smiles_test)
            ], axis=0)

        self.y_train = self.labels[self.split_df[self.split_df["split"] == "train"]["index"].values]
        self.y_test = self.labels[self.split_df[self.split_df["split"] == "test"]["index"].values]


        self.x_train = torch.from_numpy(self.x_train).int()
        self.x_test = torch.from_numpy(self.x_test).int()
        self.y_train = torch.from_numpy(self.y_train).int()
        self.y_test = torch.from_numpy(self.y_test).int()

    def get_train_test_splits(self):

        return self.x_train, self.x_test, self.y_train, self.y_test


    def __len__(self):

        return self.x.shape[0]
    
    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]






def compute_splits(split_path:Path, 
                   random_state:int, 
                   split_type:str, 
                   df:pd.DataFrame, 
                   smiles_col:str):

    # reset the index of the dataframe
    df = df.reset_index()


    split_df = None    
    if not split_path.exists():
            print(f"computing split file: {split_path}")
            if split_type == "random":
                train_idxs, test_idxs = train_test_split(
                    list(range(len(df))), random_state=random_state
                )

            elif split_type == "scaffold":

                scaffoldsplitter = dc.splits.ScaffoldSplitter()
                idxs = np.array(list(range(len(df))))

                dataset = dc.data.DiskDataset.from_numpy(
                    X=idxs, w=np.zeros(len(df)), ids=df[smiles_col]
                )
                train_data, test_data = scaffoldsplitter.train_test_split(dataset)

                train_idxs = train_data.X
                test_idxs = test_data.X


            # import ipdb
            # ipdb.set_trace()
            # create the train/test column
            train_df = df.loc[train_idxs]
            test_df = df.loc[test_idxs]
            train_df["split"] = ["train"] * len(train_df)
            test_df["split"] = ["test"] * len(test_df)

            split_df = pd.concat([train_df, test_df])
            split_df.to_csv(split_path, index=True)

    else:
        print(f"split path: {split_path} exists. loading.")
        split_df = pd.read_csv(split_path, index_col=0)

    return split_df
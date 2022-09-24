from base64 import encode
from cProfile import label
from enum import unique
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smiles', nargs='*', default=['CC[N+](C)(C)Cc1ccccc1Br'])
parser.add_argument('--ngram-order', type=int, default=1)
parser.add_argument('--tokenizer', choices=["atomwise", "ngram"])
parser.add_argument('--D', type=int, help="size of encoding dimension", default=10000)
parser.add_argument('--dataset', choices=["bbbp", "sider", "clintox", "dude"], required=True)
args = parser.parse_args()

import torch
from hdpy.fsl.classification_modules import HDModel
from tqdm import tqdm 



class MoleHDEncoder(HDModel):

    def __init__(self, D):
        super(MoleHDEncoder, self).__init__()

        # "D" is the dimension of the encoded representation
        self.D = D

    def build_item_memory(self, dataset_tokens):
        self.item_mem = {}


        # import ipdb
        # ipdb.set_trace()
        print("building item memory")
        for tokens in tqdm(dataset_tokens):

            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = torch.bernoulli(torch.empty(self.D).uniform_(0,1))
                    token_hv = torch.where(token_hv > 0 , token_hv, -1) 
                    self.item_mem[token] = token_hv

        print(f"item memory formed with {len(self.item_mem.keys())} entries.")

    def encode(self, tokens):

        # tokens is a list of tokens, i.e. it corresponds to 1 sample

        hv = torch.zeros(self.D)

        for idx, token in enumerate(tokens):
            token_hv = self.item_mem[token]
            hv = hv + torch.roll(token_hv, idx)


        # binarize
        hv = torch.where(hv > 0, hv, -1)
        hv = torch.where(hv <= 0, hv, 1)
        return hv

    def encode_dataset(self, dataset_tokens):
        dataset_hvs = []
        for tokens in tqdm(dataset_tokens):
            dataset_hvs.append(self.encode(tokens))


        # # import ipdb
        # ipdb.set_trace()
        return dataset_hvs

    def build_am(self, dataset_hvs, labels):



        # import ipdb 
        # ipdb.set_trace()

        self.am = {}

        for hv, label in zip(dataset_hvs, labels):


            if int(label) not in self.am.keys():
                self.am[int(label)] = hv
            else:
                self.am[int(label)] += hv

    def retrain(self, dataset_hvs, labels):


        # import ipdb
        # ipdb.set_trace()

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        for hv, label in tqdm(zip(dataset_hvs, labels), desc=f"retraining...", total=len(dataset_hvs)):

            
            out = int(torch.argmax(torch.nn.CosineSimilarity()(am_array, hv)))

            if out == int(label):
                pass
            else:
                self.am[out] -= hv
                self.am[int(label)] += hv

    def predict(self, dataset_hvs):

        
        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)

        pred_list = []
        for hv in tqdm(dataset_hvs, total=len(dataset_hvs), desc="predicting..."):


            out = int(torch.argmax(torch.nn.CosineSimilarity()(am_array, hv)))
            pred_list.append(out)
        

        return pred_list

    def compute_confidence(self, dataset_hvs):

        # because we'll use this multiple times but only need to compute once, taking care to maintain sorted order 
        am_array = torch.concat([self.am[key].reshape(1,-1) for key in sorted(self.am.keys())], dim=0)


        eta_list = []

        for hv in dataset_hvs:
            out = torch.nn.CosineSimilarity()(am_array, hv)
            eta = (1/2) + (1/4) * (out[1] - out[0])

            eta_list.append(eta)

        return eta_list


def train(model, hv_train, y_train, epochs=10):

    model.build_am(hv_train, y_train)

    for _ in range(epochs):
        model.retrain(hv_train, y_train)

    return model


def test(model, hv_test, y_test): 

    task_pred_list = []
    task_eta_list = []

    pred_list = model.predict(hv_test)
    eta_list = model.compute_confidence(hv_test)

    return {"y_pred": pred_list, "y_true": y_test, "eta": eta_list}







def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def main(smiles, labels):

    import multiprocessing as mp

    tokenizer_func = None

    import functools


    if args.tokenizer == "atomwise":

        tokenizer_func = atomwise_tokenizer

    elif args.tokenizer == "ngram":
        tokenizer_func = functools.partial(kmer_tokenizer, ngram=args.ngram_order)


    with mp.Pool(mp.cpu_count()-2) as p:
        toks = list(tqdm(p.imap(tokenizer_func, smiles), total=len(smiles)))

    hd_model = MoleHDEncoder(D=args.D)
    hd_model.build_item_memory(toks)

    ##########################################################################################






    # print(hd_model.item_mem)


    dataset_hvs = hd_model.encode_dataset(toks)
    dataset_labels = labels 




    # import ipdb 
    # ipdb.set_trace()


    from sklearn.model_selection import train_test_split


    train_idxs, test_idxs = train_test_split(list(range(len(dataset_labels))))

    dataset_hvs_train = [dataset_hvs[idx] for idx in train_idxs]
    dataset_labels_train = [dataset_labels[idx] for idx in train_idxs]


    dataset_hvs_test = [dataset_hvs[idx] for idx in test_idxs]
    dataset_labels_test = [dataset_labels[idx] for idx in test_idxs]



    train(hd_model, dataset_hvs_train, dataset_labels_train)

    result_dict = test(hd_model, dataset_hvs_test, dataset_labels_test)

    from sklearn.metrics import classification_report


    # import ipdb 
    # ipdb.set_trace()

    import numpy as np
    from sklearn.metrics import roc_auc_score

    # task_pred_array = np.array(result_dict["y_pred"][task_idx]).squeeze()
    task_pred_array = np.array(result_dict["y_pred"])
    task_eta_array = np.array(result_dict["eta"]).reshape(-1,1)
    task_true_array = np.array(result_dict["y_true"])

    print(classification_report(y_pred=task_pred_array, y_true=task_true_array))
    print(f"roc-auc: {roc_auc_score(y_score=task_eta_array, y_true=np.array(dataset_labels_test))}")

    # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1

    # if n_tasks == 1:

if __name__ == "__main__":

    '''
        each problem (bbbp, sider, clintox) have n binary tasks..we'll form a separate AM for each
    '''

    import pandas as pd


    if args.dataset == "bbbp":

        df = pd.read_csv('/usr/workspace/bcwc/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv')

        smiles = df['rdkitSmiles'].values.tolist()
        labels = df['p_np'].values.reshape(-1,1)

        n_tasks = 1

        main(smiles, labels)

    elif args.dataset == "sider":

        df = pd.read_csv('/usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/mole_hd/sider.csv')

        label_cols = [x for x in df.columns.values if "smiles" not in x] 
        smiles = df['smiles'].values.tolist()
        labels = df[[x for x in df.columns.values if "smiles" not in x]].values
        n_tasks = len(label_cols)

        for task_idx in range(n_tasks):

            main(smiles, labels[:, task_idx])
    elif args.dataset == "clintox":

        df = pd.read_csv("/usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/mole_hd/clintox.csv")
        label_cols = [x for x in df.columns.values if "smiles" not in x]

        smiles = df['smiles'].values.tolist()
        labels = df[label_cols].values
        n_tasks = len(label_cols)


        # import ipdb 
        # ipdb.set_trace()

        for task_idx in range(n_tasks):
            main(smiles, labels[:, task_idx])


    elif args.dataset == "dude":

        df = pd.read_csv("src_gbsa_smiles.csv")

        smiles = df['smiles'].values.tolist()
        labels = df['decoy'].apply(lambda x: int(not x)).values.reshape(-1,1)

        n_tasks = 1

        main(smiles, labels)
    

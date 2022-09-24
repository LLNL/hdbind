from tkinter import E
import torch
import numpy as np
from tqdm import tqdm 
from hdpy.fsl.classification_modules import HDModel






class ECFPEncoder(HDModel):


    def __init__(self, D):
        super(ECFPEncoder, self).__init__()

        self.D = D 

    def build_item_memory(self, n_bits:int):

        self.item_mem = {"pos": {}, "value": {}}


        print("building item memory")

        for pos in tqdm(range(n_bits)):
            pos_hv = torch.bernoulli(torch.empty(self.D).uniform_(0,1))
            pos_hv = torch.where(pos_hv > 0, pos_hv, -1)
            self.item_mem["pos"][pos] = pos_hv
        
        for value in range(2):
            value_hv = torch.bernoulli(torch.empty(self.D).uniform_(0,1))
            value_hv = torch.where(value_hv > 0, value_hv, -1)
            self.item_mem["value"][value] = value_hv    

        
        print(f"item memory formed with {len(self.item_mem['pos'].keys())} (pos) and {len(self.item_mem['value'].keys())} entries...")


    def encode(self, datapoint):
        
        # datapoint is just a single ECFP

        hv = torch.zeros(self.D)

        for pos, value in enumerate(datapoint):

            pos_hv = self.item_mem["pos"][pos]
            value_hv = self.item_mem["value"][value]

            # bind both item memory elements? or should I use a single 2 by n_bit matrix of values randomly chosen to associate with all possibilities?
            hv = hv + (pos_hv * value_hv)

        # binarize
        hv = torch.where(hv > 0, hv, -1)
        hv = torch.where(hv <= 0, hv, 1)

        return hv


    def encode_dataset(self, dataset):

        dataset_hvs = []

        for datapoint in tqdm(dataset, desc="encoding dataset..."):
            dataset_hvs.append(self.encode(datapoint))


        return dataset_hvs

    def build_am(self, dataset_hvs, labels):


        self.am = {}

        for hv, label in zip(dataset_hvs, labels):

            if int(label) not in self.am.keys():
                self.am[int(label)] = hv
            else:
                self.am[int(label)] += hv


    def retrain(self, dataset_hvs, labels):


        # import ipdb
        # ipdb.set_trace()

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


if __name__ == "__main__":


    

    data = np.load("/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/data_utils/datasets/dude/deepchem_feats/fa10/ecfp/data.npy")    



    # data = data[:1000, :]



    hd_model = ECFPEncoder(D=10000)
    hd_model.build_item_memory(n_bits=data.shape[1] - 1)

    dataset_hvs = hd_model.encode_dataset(data[:, :-1])

    dataset_labels = 1 - data[:, -1]

    from sklearn.model_selection import train_test_split

    # import ipdb
    # ipdb.set_trace()

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

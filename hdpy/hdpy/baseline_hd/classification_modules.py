import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset
from hdpy.hdpy.hd_model import HDModel
from hdpy.hdpy.utils import CustomDataset
from hdpy.hdpy.ecfp_hd.encode import compute_fingerprint_from_smiles
import time
import numpy as np


class RPEncoder(HDModel):
    def __init__(self, input_size, D, num_classes):
        super(RPEncoder, self).__init__(D=D)
        # super()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float()*2-1
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float()


    def encode(self, x):
                
        hv = self.rp_layer(x.float())
        hv = torch.where(hv>0, 1.0, -1.0)
        return hv 

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


    def fit(self, x_train, y_train, num_epochs):

        # cross entropy likes the long tensor so just do this once before training instead of multiple times
        y_train = y_train.long()

        # Train the model

        train_dataloader = DataLoader(CustomDataset(x_train, y_train), batch_size=32)

        for batch in tqdm(train_dataloader, desc="training MLP..."):
            # Forward pass
            features, labels = batch
            self.optimizer.zero_grad()
            outputs = self.forward(features)
            loss = self.criterion(outputs, labels)
            print(loss) 
            # Backward and optimize
            loss.backward()
            self.optimizer.step()


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
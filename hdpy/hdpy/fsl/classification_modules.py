import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import ipdb 

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

    def init_class(self, x_train, train_labels):
        raise NotImplementedError("Please use a subclass of HDModel.")

    def encode(self, x):
        raise NotImplementedError("Please use a subclass of HDModel.") 

    def predict(self, x):
        if x is None:
            raise NotImplementedError("Please use a subclass of HDModel.")
        else:
            enc_hvs = self.encode(x)
            # out = nn.CosineSimilarity()(self.am, enc_hv)
            # return out
            # preds = torch.zeros(x.shape[0])
            # for i in tqdm(range(enc_hvs.size()[0])):
                # sims = nn.CosineSimilarity()(self.am, enc_hvs[i].unsqueeze(dim=0)).unsqueeze(dim=0)
        
                # preds[i] = torch.argmax(sims, dim=1)
            preds = torch.argmax(torchmetrics.functional.pairwise_cosine_similarity(enc_hvs, self.am), dim=1)


        return preds 

    def forward(self, x):
        # enc_hv = self.encode(x)
        # out = nn.CosineSimilarity()(self.am, enc_hv)     
        out = self.predict(x) 
        return out


    def train_step(self, train_features, train_labels, lr=1.0):
        shuffle_idx = torch.randperm(train_features.size()[0])
        train_features = train_features[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        # import ipdb
        # ipdb.set_trace()
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



        # self.am[train_labels[misclass_mask].int()] -= lr * enc_hvs[misclass_mask]

        # train_labels[misclass_mask].int() -= lr * enc_hvs[misclass_mask]

        # self.am[~train_labels[misclass_mask].int()] += lr * enc_hvs[misclass_mask]



        '''
        # serial training loop 
        for idx, predict in enumerate(preds):
            if predict != train_labels[idx]:
                self.am[predict] -= lr * enc_hvs[idx]
                self.am[train_labels[idx].int()] += lr * enc_hvs[idx] 
            
        '''

        self.am = binarize(self.am) 

    def fit(self, x_train, y_train, num_epochs, lr=1.0):
        for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
            self.train_step(train_features=x_train, train_labels=y_train, lr=lr)


class HD_Classification(HDModel):
    def __init__(self, input_size, D, num_classes):
        super(HD_Classification, self).__init__()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float()*2-1
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()

    def RP_encoding(self, x):

        # ipdb.set_trace()
        out = self.rp_layer(x)
        out = torch.where(out>0, 1.0, -1.0)
        return out

    def encode(self, x):
        return self.RP_encoding(x)

    def init_class(self, x_train, train_labels):
        self.am = self.init_class_hvs
        out = self.RP_encoding(x_train)

        # for i in tqdm(range(x_train.size()[0]), desc="encoding.."):
        #   self.init_class_hvs[train_labels[i].int()] += out[i]

        am_array = torch.index_select(self.init_class_hvs, 0, train_labels.int())
        am_array += out

        for label in train_labels.int().unique():
            self.am[int(label)] = am_array[train_labels == label].sum(dim=0)

        self.am = self.init_class_hvs
        self.am = binarize(self.am)


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
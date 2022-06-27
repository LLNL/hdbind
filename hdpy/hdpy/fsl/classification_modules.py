import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

def binarize(x):
    return torch.where(x>0, 1.0, -1.0)





class HDModel(nn.Module):

    def __init__(self):
        super(HDModel, self).__init__()
        self.am = None

    def init_class(self, x_train, labels_train):
        raise NotImplementedError("Please use a subclass of HDModel.")

    def encode(self, x):
        raise NotImplementedError("Please use a subclass of HDModel.") 

    def predict(self, enc_hv):
        if enc_hv is None:
            raise NotImplementedError("Please use a subclass of HDModel.")
        else:
            out = nn.CosineSimilarity()(self.am, enc_hv)
            return out

    def forward(self, x):
        env_hv = self.encode(x)
        out = nn.CosineSimilarity()(self.am, enc_hv)      
        return out


    def train_step(self, train_features, train_labels, lr=1.0):
        shuffle_idx = torch.randperm(train_features.size()[0])
        train_features = train_features[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        import ipdb
        ipdb.set_trace()
        enc_hvs = self.encode(train_features)
        for i in tqdm(range(enc_hvs.size()[0])):
            sims = nn.CosineSimilarity()(self.am, enc_hvs[i].unsqueeze(dim=0))
            
            predict = torch.argmax(sims, dim=1)

            if predict != train_labels[i]:
                self.am[predict] -= lr * enc_hvs[i]
                self.am[train_labels[i]] += lr * enc_hvs[i] 


class HD_Classification(HDModel):
    def __init__(self, input_size, D, num_classes):
        super(HD_Classification, self).__init__()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        init_rp_mat = torch.bernoulli(torch.tensor([[0.5] * input_size] * D)).float()*2-1
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)

        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()

    def RP_encoding(self, x):
        out = self.rp_layer(x)
        out = torch.where(out>0, 1.0, -1.0)
        return out

    def encode(self, x):
        return self.RP_encoding(x)

    def init_class(self, x_train, labels_train):
        out = self.RP_encoding(x_train)
        for i in tqdm(range(x_train.size()[0]), desc="encoding.."):
            self.init_class_hvs[labels_train[i]] += out[i]
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

    def init_class(self, x_train, labels_train):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[labels_train[i]] += out[i]
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

    def init_class(self, x_train, labels_train):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[labels_train[i]] += out[i]
        self.am = self.init_class_hvs

        self.am = binarize(self.am)


class HD_Level_Classification(nn.Module):
    def __init__(self, input_size, D, num_classes, quan_level=8):
        super(HD_Level_Classification, self).__init__()
        self.rp_layer = nn.Linear(input_size, D, bias=False)

        # self.
        self.quantize_scale = 1./quan_level

        init_rp_mat = torch.bernoulli(torch.tensor([[density] * input_size] * D)).float()
        self.rp_layer.weight = nn.parameter.Parameter(init_rp_mat, requires_grad=False)
        
        self.init_class_hvs = torch.zeros(num_classes, D).float().cuda()

    def quantize(self, x):
        return torch.fake_quantize_per_tensor_affine(features_support, scale=self.quantize_scale, zero_point=0, quant_min=0, quant_max=3)

    #def encoding(self, x):
    #    out = self.rp_layer(x)
        # out = torch.where(out>0, 1.0, -1.0)
    #    return out

    def init_class(self, x_train, labels_train):
        out = self.RP_encoding(x_train)
        for i in range(x_train.size()[0]):
            self.init_class_hvs[labels_train[i]] += out[i]
        # self.am = nn.parameter.Parameter(self.init_class_hvs, requires_grad=True)
        self.am = self.init_class_hvs

    def forward(self, x):
        out = self.level_encoding(x)
        out = nn.CosineSimilarity()(class_hvs=self.am, enc_hv=out)      
        return out



# BENCHMARK MODELS


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, net_type):
        super(ClassifierNetwork, self).__init__()
        self.net_type = net_type
        if net_type=='MLP':
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        elif net_type=='Linear':
            self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        if self.net_type=='MLP':
            out = self.tanh(out)
            out = self.fc2(out)
        return out


class kNN(nn.Module):
    def __init__(self, x_train, labels_train, num_classes:int, distance_type='L2', k=1):
        super(kNN, self).__init__()
        self.k = k
        self.distance_type = distance_type
        self.support_embeddings = x_train
        self.support_labels = labels_train
        # self.num_classes = max(labels_train)+1
        self.num_classes = num_classes

    '''
    def cosine_similarity(self, class_hvs, enc_hv):
        #class_hvs = torch.div(class_hvs, torch.norm(class_hvs, dim=1, keepdim=True))
        #enc_hv = torch.div(enc_hv, torch.norm(enc_hv, dim=1, keepdim=True))
        #return torch.matmul(enc_hv, class_hvs.t())
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(class_hvs, enc_hv)
    '''

    def forward(self, x):
        if self.distance_type in ['L1', 'L2']:
            #  [num_query, 1, embed_dims]
            query = torch.unsqueeze(x, 1)
            #  [1, num_support, embed_dims]
            support = torch.unsqueeze(self.support_embeddings, 0)

            if self.distance_type == 'L1':
                #  [num_query, num_support]
                distance = torch.linalg.norm(query-support, dim=2, ord=1)
            else:
                distance = torch.linalg.norm(query-support, dim=2, ord=2)
        elif self.distance_type == 'cosine':
            distance = -1.0 * nn.CosineSimilarity()(self.support_embeddings, x)
        else:
            raise ValueError('Distance must be one of L1, L2 or cosine.')        
             
        _, idx = torch.topk(-distance, k=self.k)
        idx = torch.squeeze(idx, dim=1)
        idx = torch.gather(self.support_labels, 0, idx)

        one_hot_classification = F.one_hot(idx.long(), num_classes=self.num_classes)
        return one_hot_classification



#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import torch.nn.functional as F
from hdpy.utils import timing_part
from hdpy.rff_hdc.utils import prepare_data
from tqdm import tqdm

from hdpy.rff_hdc.encoder import RandomFourierEncoder

from hdpy.rff_hdc.utils import HDDataset

'''
model for binary hdc
'''


class FastSign(torch.nn.Module):
    '''
    This is a fast version of the SignActivation.
    '''

    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        out_backward = torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward


class BinaryLinear(torch.nn.Linear):
    '''
    A fully connected layer with weights binarized to {-1, +1}.
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__(
            in_features, out_features, bias
        )
        self.binarize = FastSign()

    def forward(self, input):
        return F.linear(input, self.binarize(self.weight), self.bias)



class BModel(torch.nn.Module):
    def __init__(self, in_dim=32768, classes=10, data_dir=None, device=None):
        super(BModel, self).__init__()

        assert data_dir is not None 
        self.data_dir = data_dir


        self.device = device

        self.in_dim = in_dim
        self.fc = BinaryLinear(self.in_dim, classes, bias=False)




    def forward(self, x):
        x = self.fc(x) * (1.0 / self.in_dim ** 0.5)
        return x


    # def train_step(self, train_features, train_labels, lr=1.0):


    def fit(self, features, labels, num_epochs, lr=1.0, device="cuda"):

        trainloader, testloader = prepare_data(self.data_dir, batch_size=512, num_workers=16)
        
        #for _ in tqdm(range(num_epochs), total=num_epochs, desc="training HD..."):
        #    self.train_step(train_features=features, train_labels=labels, lr=lr)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(num_epochs), total=num_epochs, desc=f"training BModel"):
            # print("Epoch:", epoch + 1)
            self.train()
            
            with timing_part("TRAIN-RFF") as timer:
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    if model == 'rff-hdc':
                        outputs = self.forward(2 * inputs - 1)
                    else:
                        outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, batch_predicted = torch.max(outputs.data, 1)
         

    def predict(self, inputs, model_='rff-hdc'):


        _ , testloader = prepare_data(self.data_dir, batch_size=512, num_workers=16)

        data, labels = testloader.dataset[:]        

        self.eval()
        with torch.no_grad():
            inputs, labels = inputs.to(self.device), data.to(self.device)
            if model_ == 'rff-hdc':
                # why is this happening?
                outputs = self.forward(2 * data.cuda() - 1)
            else:
                outputs = self.forward(data.cuda())
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability


        return preds


# '''
# model for cyclic group hdc of different order
# '''

class FastRound(torch.nn.Module):
    '''
    This is a fast version of the round.
    '''

    def __init__(self):
        super(FastRound, self).__init__()

    def forward(self, input):
        out_forward = torch.round(input)
        out_backward = input  # torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward


class RoundLinear(torch.nn.Linear):
    '''
    A fully connected layer with weights rounded to closest integers
    '''

    def __init__(self, in_features, out_features, gorder, bias=True):
        super(RoundLinear, self).__init__(
            in_features, out_features, bias
        )
        self.gorder = gorder
        self.Bias = bias
        self.round = FastRound()
        self.radius = torch.nn.Parameter(torch.ones(1))  # 1.0

    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / (1.0 * self.gorder) * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        return pts

    def GroupSim(self, input, weight):
        map_weight = self.pts_map(weight, r=self.radius)
        map_input = self.pts_map(input).unsqueeze(1)
        return torch.sum(torch.sum(map_weight * map_input, dim=-1), dim=-1)

    def forward(self, input):
        weight_q = self.weight
        if not self.training:
            weight_q = self.round(self.weight)
        sims = self.GroupSim(input, weight_q)
        if self.Bias:
            sims += self.bias
        return sims


class GModel(torch.nn.Module):
    def __init__(self, gorder, in_dim=32768, classes=10, data_dir=None, device=None, enc_dim=2000, encoder=None):
        super(GModel, self).__init__()

        assert data_dir is not None 
        self.data_dir = data_dir

        assert encoder is not None 
        self.encoder = encoder 

        self.mem = encoder.build_item_mem()



        self.enc_dim=enc_dim


        self.device = device
        self.in_dim = in_dim
        self.fc = RoundLinear(self.in_dim, classes, gorder, bias=False)

    def forward(self, x):
        x = self.fc(x) * (1.0 / self.in_dim ** 0.5)
        return x


    def fit(self, x_train, y_train, num_epochs, lr=1.0, device="cuda", num_workers=4, batch_size=16):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # encoder = RandomFourierEncoder(
            # input_dim=x_train.shape[1], gamma=0.5, gorder=8, output_dim=self.enc_dim)

        #todo: do this in the main function, don't catch this in the profiling
        x_train_enc = self.encoder.encode_data(x_train)

        for epoch in tqdm(range(num_epochs), total=num_epochs, desc=f"training GModel"):
            self.train()
            with timing_part("TRAIN-RFF") as timer:

                optimizer.zero_grad()

                outputs = self.forward(torch.tensor(x_train_enc).to('cuda'))
                loss = criterion(outputs, y_train.long())
                loss.backward()
                optimizer.step()
        

    def predict(self, inputs):


        # import ipdb
        # ipdb.set_trace()

        inputs_enc = self.encoder.encode_data(inputs)

        self.eval()
        with torch.no_grad():
            inputs_enc = inputs_enc.to('cuda')
            outputs = self.forward(inputs_enc.cuda())
        preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability


        return preds

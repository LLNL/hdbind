#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
from hdpy.rff_hdc.encoder import LinearEncoder, RandomFourierEncoder
from pathlib import Path

def quantize(data, precision=8):
    # assume min and max of the data is -1 and 1
    scaling_factor = 2 ** (precision - 1) - 1
    # data = np.round(data * scaling_factor)

    data = torch.round(data * scaling_factor)
    # shift the quantized data to positive and rescale to [0, 1.0]
    # return (data + scaling_factor) / 255.0 #TODO: shouldn't be using 255

    return (data + scaling_factor) / data.shape[1] #TODO: shouldn't be using 255

def encode_and_save(data_dir:str, dataset:str, x_train:np.array, x_test:np.array, y_train:np.array, y_test:np.array, model_type:str, dim:int, gamma:float, gorder:int):
# def encode_and_save(data_dir:str, dataset:str, model:str, dim:int, gamma:float, gorder:int):
    ### load data using torch with pixel values in [0,1]

    data_dir = Path(data_dir)

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    if dataset == "dude": 

        x_train, y_train = torch.tensor(quantize(x_train, precision=8)).unsqueeze(1), torch.tensor(y_train).long()
        x_test, y_test = torch.tensor(quantize(x_test, precision=8)).unsqueeze(1), torch.tensor(y_test).long()

        trainset = HDDataset(x_train, y_train)
        testset = HDDataset(x_test, y_test)

        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    else:
        raise ValueError("Dataset is not supported.")



    assert len(trainset[0][0].size()) > 1
    channels = trainset[0][0].size(0)
    print('# of channels of data', channels)
    input_dim = torch.prod(torch.tensor(list(trainset[0][0].size())))
    print(f"input_dim: {input_dim}")
    print('# of training samples and test samples', len(trainset), len(testset))

    if model_type == 'linear-hdc':
        print("Encoding to binary HDC with linear hamming distance.")
        encoder = LinearEncoder(dim=dim)
    elif 'rff' in model_type:
        print("Encoding with random fourier features encoder.")
        encoder = RandomFourierEncoder(
            input_dim=input_dim, gamma=gamma, gorder=gorder, output_dim=dim)
    else:
        raise ValueError("No such feature type is supported.")

    mem = encoder.build_item_mem()
    print("Encoded features to hypervectors with size: ", mem.size())
    torch.save(mem, data_dir / 'item_mem.pt')

    print("Encoding training data...")
    train_hd, y_train = encoder.encode_data_extract_labels(trainset)
    torch.save(train_hd, data_dir / 'train_hd.pt')
    torch.save(y_train, data_dir / 'y_train.pt')
    del train_hd, y_train
    torch.cuda.empty_cache()  # in case of CUDA OOM

    print("Encoding test data...")
    test_hd, y_test = encoder.encode_data_extract_labels(testset)
    torch.save(test_hd, data_dir / 'test_hd.pt')
    torch.save(y_test, data_dir / 'y_test.pt')
    del test_hd, y_test
    torch.cuda.empty_cache()


def load(data_dir):
    #     mem = torch.load(f'{args.data_dir}/item_mem.pt')
    #     print("Loaded pixel hypervectors with size: ", mem.size())

    data_dir = Path(data_dir)
    print("Loading encoded training data...")
    train_hd = torch.load(data_dir / 'train_hd.pt')
    y_train = torch.load(data_dir / 'y_train.pt')

    print("Loading encoded test data...")
    test_hd = torch.load(data_dir / 'test_hd.pt')
    y_test = torch.load(data_dir / 'y_test.pt')

    print(f"Size of encoded training data {train_hd.size()} and test data {test_hd.size()}")
    return train_hd, y_train, test_hd, y_test


class HDDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def prepare_data(data_dir, batch_size=128, num_workers=1):
    train_hd, y_train, test_hd, y_test = load(data_dir)
    train_dataset = HDDataset(train_hd, y_train)
    test_dataset = HDDataset(test_hd, y_test)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=num_workers)
    return trainloader, testloader

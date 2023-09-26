import sys
import numpy as np
from utils import get_features_fewshot_single
from hdpy.hdpy.baseline_hd.classification_modules import kNN, ClassifierNetwork, HD_Classification

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm

import pandas as pd
from collections import defaultdict


model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']

datasets = ['omniglot', 'aircraft', 'cu_birds', 
            'dtd', 'quickdraw', 'fungi',
            'vgg_flower', 'traffic_sign', 'cifar-100'
            'imagenet-super', 'imagenet-vehicle']

dataset_path = {
    'omniglot':'data/transferred_features_omniglot-py', 
    'aircraft':None, 
    'cu_birds':'data/transferred_features_CUB-200/images',
    'dtd':'data/transferred_features_dtd/images', 
    'quickdraw':None, 
    'fungi':None,
    'vgg_flower':None, 
    'traffic_sign':'data/transferred_features_traffic-sign/Images',
    'cifar-100':'data/transferred_features_cifar-100',
    'imagenet-super':'data/transferred_features_imagenet-super', 
    'imagenet-vehicle':'data/transferred_features_imagenet-vehicle'
}

# Device configuration
device = torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")


def train_model(model, features, labels, criterion, optimizer,
                num_epochs=50):
    # Train the model
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        if not args.nol2:
            c = torch.tensor(args.gamma, device=device)
            l2_reg = torch.tensor(0., device=device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param)

            loss += c * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # print(model.am.grad)
        optimizer.step()

        # print('Epoch [{}/{}],  Loss: {:.4f}'
        #     .format(epoch + 1, num_epochs, loss.item()))



def train_HD(model, train_features, train_labels, test_features, test_labels, epochs, use_cuda=True):
    acc_max = -np.inf
    for epoch in range(epochs):
        acc_test = test_model(model, test_features, test_labels)
        # sys.stdout.write('\nepoch %s: ' %epoch)
        # sys.stdout.write('%.4f ' % acc_test)
        # sys.stdout.flush()
        # print('')

        model.train_step(train_features, train_labels)



def test_model(model, features, labels):
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted==y).sum().item()

    return 100 * correct / total


def main(args, result_dict):
    data = args.datapath
    model_name = args.model
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size

    data_path = os.path.join(data, '')

    meta_folder = os.path.join(data_path, model_name)

    folders = [os.path.join(meta_folder, label) \
               for label in os.listdir(meta_folder) \
               if os.path.isdir(os.path.join(meta_folder, label)) \
               ]

    accs = []
    for i in range(n_problems):
        sampled_folders = random.sample(folders, nway)

        features_support, labels_support, \
        features_query, labels_query = get_features_fewshot_single(nb_shot=kshot, 
                                                                   paths=sampled_folders,
                                                                   labels=range(nway), 
                                                                   nb_samples=n_img, 
                                                                   shuffle=True)

        features_support = torch.tensor(features_support, dtype=torch.float32, device=device)
        labels_support = torch.tensor(labels_support, dtype=torch.long, device=device)

        features_query = torch.tensor(features_query, dtype=torch.float32, device=device)
        labels_query = torch.tensor(labels_query, dtype=torch.long, device=device)

        features_support = torch.fake_quantize_per_tensor_affine(features_support, scale=0.5, zero_point=0, quant_min=0, quant_max=3)
        features_query = torch.fake_quantize_per_tensor_affine(features_query, scale=0.5, zero_point=0, quant_min=0, quant_max=3)

        input_size = features_support.shape[1]
        # print(torch.mean(features_support))

        if args.classifier_type == 'MLP':
            model = ClassifierNetwork(input_size, hidden_size, nway, args.classifier_type).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            train_model(model, features_support, labels_support, criterion, optimizer, num_epochs)
        elif args.classifier_type == 'HD':
            model = HD_Classification(input_size=input_size, D=hidden_size, num_classes=nway).to(device)
            model.init_class(x_train=features_support, labels_train=labels_support)
            train_HD(model, features_support, labels_support, features_query, labels_query, num_epochs)
        elif args.classifier_type in ['L1', 'L2', 'cosine']:
            # kNN test
            model = kNN(x_train=features_support, labels_train=labels_support, distance_type=args.classifier_type, k=1)
        
        accuracy_test = test_model(model, features_query, labels_query)

        # print(round(accuracy_test, 2))
        accs.append(accuracy_test)

    stds = np.std(accs)
    acc_avg = round(np.mean(accs), 2)
    ci95 = round(1.96 * stds / np.sqrt(n_problems), 2)

    result_dict['acc'].append(acc_avg)
    result_dict['ci95'].append(ci95)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Single Library Classifier')

    #parser.add_argument('--dataset', default='dtd',
    #    choices=datasets, help='evaluated dataset')
    parser.add_argument('--out_csv', required=True, help='path to output CSV file')

    parser.add_argument('--model', default='resnet18',
        choices=model_names, help='model architecture')
    parser.add_argument('--nway', default=5, type=int,
        help='number of classes')
    parser.add_argument('--kshot', default=5, type=int,
        help='number of shots (support images per class)')
    parser.add_argument('--kquery', default=15, type=int,
        help='number of query images per class')
    parser.add_argument('--n_problems', default=600, type=int,
        help='number of test problems')

    parser.add_argument('--classifier_type',
        help='set for embedding classifier: MLP, Linear, HD, L1, L2, cosine')
    parser.add_argument('--hidden_size', default=512, type=int,
        help='hidden layer size')
    parser.add_argument('--num_epochs', default=100, type=int,
        help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float,
        help='learning rate')
    parser.add_argument('--gamma', default=0.5, type=float,
        help='L2 regularization constant')

    parser.add_argument('--nol2', action='store_true', default=False,
        help='set for No L2 regularization, otherwise use L2')

    parser.add_argument('--gpu', default=0, type=int,
        help='GPU id to use.')
    parser.add_argument('--datapath', help='path to dataset')

    args = parser.parse_args()

    results = defaultdict(list)

    for dataset_i in tqdm(['dtd', 'cu_birds', 'cifar-100', 'traffic_sign']):
    # for dataset_i in tqdm(['imagenet-super', 'imagenet-vehicle']):
        for nway in [5, 10]:
            for kshot in [1, 5]:
                # for classifier_type in ['MLP', 'HD', 'L1', 'cosine']:
                for classifier_type in ['HD', 'L1', 'cosine']:
                    print(dataset_i, nway, kshot, classifier_type)
                    if classifier_type=='MLP':
                        hidden_size = 512
                        num_epochs = 100
                    elif classifier_type=='HD':
                        hidden_size = 2048
                        num_epochs = 0
                    else:
                        hidden_size = 0
                        num_epochs = 0

                    args.dataset = dataset_i
                    args.datapath = dataset_path[args.dataset]
                    args.nway, args.kshot = nway, kshot
                    args.classifier_type = classifier_type
                    args.hidden_size, args.num_epochs = hidden_size, num_epochs
                    
                    results['dataset'].append(args.dataset)
                    results['model'].append(args.model)
                    results['nway'].append(args.nway)
                    results['kshot'].append(args.kshot)
                    results['kquery'].append(args.kquery)
                    results['n_problems'].append(args.n_problems)

                    results['classifier_type'].append(args.classifier_type)
                    results['hidden_size'].append(args.hidden_size)
                    results['num_epochs'].append(args.num_epochs)
                    
                    main(args, results)
                    # print(results)
    
    pd.DataFrame(results).to_csv(args.out_csv, index=False, sep='\t')
    
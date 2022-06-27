import sys
import numpy as np
from utils import load_features 
from classification_modules import *
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score, precision_score
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from tqdm import tqdm

import pandas as pd
from collections import defaultdict

from hdpy.utils import timing_part


def get_args():

    parser = argparse.ArgumentParser(description='Single Library Classifier')

    parser.add_argument('--dataset', help='evaluated dataset (pdbbind, postera, dud-e) that determines how measurements are loaded into classes', required=True)
    # parser.add_argument('--out-csv', required=True, help='path to output CSV file')

    # parser.add_argument('--model-list', nargs='+', required=True, help="Please select from ['HD', 'HD-Sparse', 'L1', 'L2', 'cosine', 'MLP', 'Uniform', 'Majority']")
    parser.add_argument('--support-path-list', required=True, nargs='+', help='path to support data files')
    # parser.add_argument('--query-path-list', required=True, nargs='+', help='path to query data files')
    parser.add_argument('--n-problems', default=600, type=int,
        help='number of test problems')
    parser.add_argument('--hidden-size', default=512, type=int,
        help='hidden layer size')
    parser.add_argument('--num-epochs', default=100, type=int,
        help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float,
        help='learning rate')
    parser.add_argument('--gamma', default=0.5, type=float,
        help='L2 regularization constant')
    parser.add_argument('--no-l2', action='store_true', default=False,
        help='set for No L2 regularization, otherwise use L2')
    parser.add_argument('--gpu', default=0, type=int,
        help='GPU id to use.')
    parser.add_argument('--datapath', help='path to dataset')

    args = parser.parse_args()

    return args




'''
def train_model(model, features, labels, criterion, optimizer,
                num_epochs=50):
    # Train the model
    x = features.clone().detach()
    y = labels.clone().detach()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        # if not args.nol2:
            # c = torch.tensor(args.gamma, device=device)
            # l2_reg = torch.tensor(0., device=device)
            # for name, param in model.named_parameters():
                # if 'weight' in name:
                    # l2_reg += torch.norm(param)

            # loss += c * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''


'''
def train_HD(model_name, model, train_features, train_labels, test_features, test_labels, epochs, use_cuda=True):
    for epoch in range(epochs):

        model.train_step(train_features, train_labels)
'''

'''
def test_model(model_name, model, features, labels, dummy=False):


    encode_test_time = 0
    pred_test_time = 0

    if dummy:
        # if dummy, we're using an sklearn.dummy.DummmyClassifier so need to have different code...
        outputs = model.predict(features)

        return float(np.sum(outputs == labels)/outputs.shape[0]), recall_score(y_true=labels, y_pred=outputs), encode_test_time, pred_test_time

    elif model_name in ['L1', 'L2', 'cosine', 'MLP', 'Uniform']:
        x = features.clone().detach()
        y = labels.clone().detach()
        with torch.no_grad():
            correct = 0
            total = 0

            with timing_part('TEST-STEP') as timer:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)

            pred_test_time = timer.total_time

            total += y.size(0)
            correct += (predicted==y).sum().item()

        return correct / total, recall_score(y_true=y.cpu().numpy(), y_pred=predicted.cpu().numpy()), encode_test_time, pred_test_time

    else:
        # x = torch.tensor(features, dtype=torch.float32, device=device)

        x = features.clone().detach() 
        y = labels.clone().detach()
        with torch.no_grad():
            correct = 0
            total = 0
            with timing_part('ENCODE-TEST') as timer:
                x_enc = model.RP_encoding(x)
            encode_test_time = timer.total_time 

            with timing_part('TEST-STEP') as timer:
                outputs = model.predict(x_enc)
                _, predicted = torch.max(outputs.data, 1)

            pred_test_time = timer.total_time

            total += y.size(0)
            correct += (predicted==y).sum().item()

        return correct / total, recall_score(y_true=y.cpu().numpy(), y_pred=predicted.cpu().numpy()), encode_test_time, pred_test_time


'''

'''
def main(args, result_dict):
    data = args.datapath
    #model_name = args.model
    dataset = args.dataset
    n_problems = args.n_problems
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size


    dummy = False
    if args.classifier_type in ["Uniform", "Majority"]:
        dummy = True



    accs = []
    recall_scores = []
    train_time = []
    test_time = []
    train_encode_time = []
    test_encode_time = []




    for i in range(n_problems): 
    
        input_size = features_support.shape[1]

        model = None 

        if args.classifier_type == 'MLP':
            model = ClassifierNetwork(input_size, hidden_size, n_classes, args.classifier_type).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            with timing_part("MODEL-TRAIN") as timer:
                train_model(model, features_support, labels_support, criterion, optimizer, num_epochs)
            train_time.append(timer.total_time)
            train_encode_time.append(0)

        elif args.classifier_type == 'HD':
            model = HD_Classification(input_size=input_size, D=hidden_size, num_classes=n_classes).to(device)

            with timing_part("ENCODING") as timer:

                model.init_class(x_train=features_support, labels_train=labels_support)
            train_encode_time.append(timer.total_time)

            with timing_part("MODEL-TRAIN") as timer:
                train_HD(args.classifier_type, model, features_support, labels_support, features_query, labels_query, num_epochs)
            train_time.append(timer.total_time)

        elif args.classifier_type == 'HD-Kron':
            model = HD_Kron_Classification(input_size=input_size, D=hidden_size, Kron_shape=[64,32], num_classes=n_classes).to(device)

            with timing_part("ENCODING") as timer:
                model.init_class(x_train=features_support, labels_train=labels_support)
            train_encode_time.append(timer.total_time)

            with timing_part("MODEL-TRAIN") as timer:
                train_HD(args.classifier_type,model, features_support, labels_support, features_query, labels_query, num_epochs)
            train_time.append(timer.total_time)

        elif args.classifier_type == 'HD-Sparse':
            model = HD_Sparse_Classification(input_size=input_size, D=hidden_size, density=0.02, num_classes=n_classes).to(device)


            with timing_part("ENCODING") as timer:
                model.init_class(x_train=features_support, labels_train=labels_support)
            train_encode_time.append(timer.total_time)

            with timing_part("MODEL-TRAIN") as timer:
                train_HD(args.classifier_type,model, features_support, labels_support, features_query, labels_query, num_epochs)
            train_time.append(timer.total_time)

        elif args.classifier_type in ['L1', 'L2', 'cosine']:
            
            model = kNN(x_train=features_support, labels_train=labels_support, num_classes=n_classes, distance_type=args.classifier_type, k=1)
            train_time.append(0)
            train_encode_time.append(0)

        elif args.classifier_type == "Uniform":
            model = DummyClassifier(strategy="uniform")
            model.fit(features_support.cpu().numpy(), labels_support.cpu().numpy())
            train_time.append(0)
            train_encode_time.append(0)

        elif args.classifier_type == "Majority": 
            model = DummyClassifier(strategy="most_frequent")
            model.fit(features_support.cpu().numpy(), labels_support.cpu().numpy())
            train_time.append(0)
            train_encode_time.append(0)
 
        accuracy_test = 0
        recall_test = 0
        encode_time_test = 0 
        pred_time_test = 0
        if dummy:
            accuracy_test, recall_test, encode_time_test, pred_time_test = test_model(args.classifier_type, model, features_query.cpu().numpy(), labels_query.cpu().numpy(), dummy=True)

        else:
            accuracy_test, recall_test, encode_time_test, pred_time_test = test_model(args.classifier_type, model, features_query, labels_query, dummy=False)


        # return correct / total, recall_score(y_true=y.cpu().numpy(), y_pred=predicted.cpu().numpy()), encode_test_time, pred_test_time
        test_encode_time.append(encode_time_test)
        test_time.append(pred_time_test)
        accs.append(accuracy_test)
        recall_scores.append(recall_test)

    acc_stds = np.std(accs)
    acc_avg = np.mean(accs)
    acc_ci95 = 1.96 * acc_stds / np.sqrt(n_problems)
    result_dict['acc'].append(acc_avg)
    result_dict['acc_ci95'].append(acc_ci95)


    recall_stds = np.std(recall_scores)
    recall_avg = np.mean(recall_scores)
    recall_ci95 = 1.96 * recall_stds / np.sqrt(n_problems)
    result_dict['recall'].append(recall_avg)
    result_dict['recall_ci95'].append(recall_ci95)


    train_encode_stds = np.std(train_encode_time)
    train_encode_avg = np.mean(train_encode_time)
    train_encode_ci95 = 1.96 * train_encode_stds / np.sqrt(n_problems)
    result_dict['train_encode_time'].append(train_encode_avg)
    result_dict['train_encode_ci95'].append(train_encode_ci95)

    test_encode_stds = np.std(test_encode_time)
    test_encode_avg = np.mean(test_encode_time)
    test_encode_ci95 = 1.96 * test_encode_stds / np.sqrt(n_problems)
    result_dict['test_encode_time'].append(test_encode_avg)
    result_dict['test_encode_ci95'].append(test_encode_ci95)



    train_time_stds = np.std(train_time)
    train_time_avg = np.mean(train_time)
    train_time_ci95 = 1.96 * train_time_stds / np.sqrt(n_problems)
    result_dict['train_time'].append(train_time_avg)
    result_dict['train_time_ci95'].append(train_time_ci95)


    test_time_stds = np.std(test_time)
    test_time_avg = np.mean(test_time)
    test_time_ci95 = 1.96 * test_time_stds / np.sqrt(n_problems)
    result_dict['test_time'].append(test_time_avg)
    result_dict['test_time_ci95'].append(test_time_ci95)

    result_dict['support_size'].append(labels_support.shape[0])
    result_dict['query_size'].append(labels_query.shape[0])
'''


if __name__=='__main__':
    # Device configuration
    device = torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")
    args = get_args() 

    results = defaultdict(list)

    n_classes = 2



    #loading data

    

    for support_path in args.support_path_list:
        print(support_path)


    # features_support, labels_support, features_query, labels_query = load_features(support_path_list=args.support_path_list, query_path_list=args.query_path_list, dataset=args.dataset)

    # features_support = torch.tensor(features_support, dtype=torch.float32, device=device)
    # features_query = torch.tensor(features_query, dtype=torch.float32, device=device)


    # labels_support = torch.tensor(labels_support, dtype=torch.long, device=device) 
    # labels_query = torch.tensor(labels_query, dtype=torch.long, device=device) 


    # print(features_support.shape, labels_support.shape, features_query.shape, labels_query.shape)




    #training model




    # for num_epochs in [0,1,5]:


    '''
    for num_epochs in [0,1]:

        # for classifier_type in ['HD', 'HD-Sparse', 'HD-Kron','L1', 'L2', 'cosine', 'MLP', 'Uniform']:

        # for classifier_type in ['HD', 'HD-Sparse', 'L1', 'L2', 'cosine', 'MLP', 'Uniform', 'Majority']:

        for classifier_type in args.model_list:

            print(f"dataset: {args.dataset}\tnum_epochs: {num_epochs}\tclassifier_type: {classifier_type}")

            if classifier_type=='MLP':
                hidden_size = 512
                # num_epochs = num_epochs
            elif classifier_type in ['HD', 'HD-Kron', 'HD-Sparse']:
                # hidden_size = 8192
                hidden_size = 512
                # num_epochs = num_epochs
            else:
                hidden_size = 0
                # num_epochs = 0


            args.classifier_type = classifier_type
            args.hidden_size, args.num_epochs = hidden_size, num_epochs

            results['dataset'].append(args.dataset)
            results['n_problems'].append(args.n_problems)
            results['classifier_type'].append(args.classifier_type)
            results['hidden_size'].append(args.hidden_size)
            results['num_epochs'].append(args.num_epochs)


            try:
                main(args, results)
                results['success'].append(True)
            except RuntimeError as e:
                print(e)
                results['train_time'].append(0)
                results['train_time_ci95'].append(0)
                results['test_time'].append(0)
                results['test_time_ci95'].append(0)
                results['acc'].append(0)
                results['acc_ci95'].append(0)
                results['recall'].append(0)
                results['recall_ci95'].append(0)
                results['success'].append(False)

                results['train_encode_time'].append(0)
                results['train_encode_ci95'].append(0)
                results['test_encode_time'].append(0)
                results['test_encode_ci95'].append(0)
                results['support_size'].append(0)
                results['query_size'].append(0)

            except ValueError as e:
                print(e)
                results['train_time'].append(0)
                results['train_time_ci95'].append(0)
                results['test_time'].append(0)
                results['test_time_ci95'].append(0)
                results['acc'].append(0)
                results['acc_ci95'].append(0)
                results['recall'].append(0)
                results['recall_ci95'].append(0)
                results['success'].append(False)

                results['train_encode_time'].append(0)
                results['train_encode_ci95'].append(0)
                results['test_encode_time'].append(0)
                results['test_encode_ci95'].append(0)
                results['support_size'].append(0)
                results['query_size'].append(0)


    output_path = Path(f"{args.out_csv}")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False, sep='\t')
    '''
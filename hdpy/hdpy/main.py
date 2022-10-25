from tkinter import W
import numpy as np
from hdpy.fsl.utils import load_features
# TODO: replace the catch-all import 
from hdpy.hdpy.baseline_hd.classification_modules import *
from hdpy.rff_hdc.model import BModel, GModel
from hdpy.rff_hdc.utils import encode_and_save
from sklearn.dummy import DummyClassifier
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import torchmetrics
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from hdpy.utils import timing_part


from collections import defaultdict
def get_args():

    parser = argparse.ArgumentParser(description='Single Library Classifier')

    parser.add_argument('--dataset', help='evaluated dataset (pdbbind, postera, dude) that determines how measurements are loaded into classes', required=True)
    parser.add_argument('--out-csv', required=True, help='path to output CSV file')

    parser.add_argument('--model-list', nargs='+', required=True, help="Please select from ['rff-hdc', 'rff-gvsa', 'HD', 'HD-Sparse', 'HD-Level', 'L1', 'L2', 'cosine', 'MLP', 'Uniform', 'Majority']")
    parser.add_argument('--train-path-list', required=True, nargs='+', help='path to train data files')
    parser.add_argument('--test-path-list', nargs='+', help='path to test data files')
    parser.add_argument('--n-problems', default=600, type=int,
        help='number of test problems')
    parser.add_argument('--hidden-size', default=512, type=int,
        help='hidden layer size')
    parser.add_argument('--num-epochs', default=100, type=int,
        help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
        help='learning rate')
    parser.add_argument('--gamma', default=0.5, type=float,
        help='L2 regularization constant')
    parser.add_argument('--no-l2', action='store_true', default=False,
        help='set for No L2 regularization, otherwise use L2')
    parser.add_argument('--gpu', default=0, type=int,
        help='GPU id to use.')
    parser.add_argument('--out-data-dir', help='not sure what this is doing actually')
    parser.add_argument('--gorder', type=int, default=8, help="order of the cyclic group required for G-VSA")
    args = parser.parse_args()

    return args




def load_data_list(data_path_list, dataset):

    features_list = []
    labels_list = []

    for path in data_path_list:
        print(f"loading {path}...")
        features, labels = load_features(path=path, dataset=dataset)
        features_list.append(features)
        labels_list.append(labels)
    
    
    features = np.concatenate(features_list, dtype=np.float32)
    labels = np.concatenate(labels_list, dtype=np.float32)
    return features, labels



def load_data(args):    
 

    x_train, y_train, x_test, y_test = None, None, None, None

    # use this if you don't have precomputed train/test splits
    if args.test_path_list is None:
        features, labels = load_data_list(args.train_path_list)
        x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.2, stratify=labels.cpu())

    # use this in the other case that you do
    else:

        x_train, y_train = load_data_list(args.train_path_list, dataset=args.dataset)
        x_test, y_test = load_data_list(args.test_path_list, dataset=args.dataset)


    scaler = StandardScaler(with_mean=True, with_std=True)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # import ipdb
    # ipdb.set_trace()

    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device) 
    y_train = torch.tensor(y_train, dtype=torch.int, device=device)
    y_test = torch.tensor(y_test, dtype=torch.int, device=device)

    return x_train, x_test, y_train, y_test

def init_model(model_type, args, x_train, y_train, x_test, y_test):

    # import ipdb
    # ipdb.set_trace()

    encode_time = -1

    if model_type == 'MLP':
        model = ClassifierNetwork(input_size=x_train.shape[1], 
                        hidden_size=512, num_classes=2, lr=args.lr).to(device)




    elif model_type == "rff-hdc":
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # criterion = torch.nn.NLLLoss()
        # channels = 3 if args.dataset == 'cifar' else 1
        channels = 1 
        if args.dataset == 'dude':
            classes = 2
        else:
            raise NotImplementedError 
        
        model = BModel(in_dim=channels * args.hidden_size, classes=classes, data_dir=args.out_data_dir, device=device).to(device)

        #trainloader, testloader = prepare_data(args)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


        with timing_part("ENCODING") as timer:
            # encode_and_save(model_type=model_type, data_dir=args.out_data_dir, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, dataset=args.dataset, dim=args.hidden_size, gamma=args.gamma, gorder=args.gorder)
            pass
        encode_time = timer.total_time 


    elif model_type == "rff-gvsa":



        channels = 1
        if args.dataset == "dude":
            classes = 2
        else:
            raise NotImplementedError 

        from hdpy.rff_hdc.encoder import RandomFourierEncoder
        encoder = RandomFourierEncoder(
            input_dim=x_train.shape[1], gamma=args.gamma, gorder=args.gorder, output_dim=args.hidden_size)
        
        model = GModel(encoder=encoder, enc_dim=args.hidden_size, gorder=args.gorder, in_dim=channels * args.hidden_size, classes=classes, data_dir=args.out_data_dir, device=device).to(device)
        model.to('cuda')








        with timing_part("ENCODING") as timer:
            # encode_and_save(model_type=model_type, data_dir=args.out_data_dir, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, dataset=args.dataset, dim=args.hidden_size, gamma=args.gamma, gorder=args.gorder)
            pass
        encode_time = timer.total_time 



    elif model_type == 'HD':
        model = HD_Classification(input_size=x_train.shape[1], D=args.hidden_size, num_classes=2).to(device)
        
        with timing_part("ENCODING") as timer:
            model.init_class(x_train=x_train, train_labels=y_train)

        encode_time = timer.total_time

    elif model_type == 'HD-Level':
        model = HD_Level_Classification(input_size=x_train.shape[1], D=args.hidden_size, num_classes=2).to(device)
        
        with timing_part("ENCODING") as timer:
            model.init_class(x_train=x_train, train_labels=y_train)

        encode_time = timer.total_time


    elif model_type == 'HD-Sparse':
        model = HD_Sparse_Classification(input_size=x_train.shape[1],
                             D=args.hidden_size, density=0.02, 
                            num_classes=2).to(device)

        with timing_part("ENCODING") as timer:
            model.init_class(features=x_train, labels=y_train)

        encode_time = timer.total_time

    elif model_type in ['L1', 'L2', 'cosine']:

        model = kNN(model_type=model_type)

    elif model_type == "Uniform":
        model = DummyClassifier(strategy="uniform")


    elif model_type == "Majority": 
        model = DummyClassifier(strategy="most_frequent")

    else:
        raise NotImplementedError("please specify valid model type")
    return model, encode_time


def train_model(model, features, labels, num_epochs, lr=1):


    # import ipdb
    # ipdb.set_trace()



    train_time = -1

    with timing_part("MODEL-TRAIN") as timer:
        model.fit(x_train=features, y_train=labels, num_epochs=num_epochs, lr=lr)

    train_time = timer.total_time

    return model, train_time


def test_model(model, features):

    pred_test_time = 0

    with torch.no_grad():

        with timing_part('TEST-STEP') as timer:
            outputs = model.predict(features)

        pred_test_time = timer.total_time

        return outputs, pred_test_time



    

def compute_metrics(y_pred, y_true):

    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    y_pred = y_pred.cpu().float()
    y_true = y_true.cpu()



    return {"acc": torchmetrics.functional.accuracy(y_pred, y_true),
        #  "auc": torchmetrics.functional.auc(y_pred, y_true, reorder=True),
         "recall-macro": torchmetrics.functional.recall(y_pred, y_true, average="macro", num_classes=2, multiclass=True),
         "recall-micro": torchmetrics.functional.recall(y_pred, y_true, average="micro", num_classes=2, multiclass=True),
         "precision-macro": torchmetrics.functional.precision(y_pred, y_true, average="macro", num_classes=2, multiclass=True),
         "precision-micro": torchmetrics.functional.precision(y_pred, y_true, average="micro", num_classes=2, multiclass=True)
        }


def print_metrics(metric_dict):
    out_str = "".join([f"{key}: {value:0.4f}\n" for key,value in metric_dict.items()])

    print(out_str)


def main():


    #loading data
 
    x_train, x_test, y_train, y_test = load_data(args)



    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train.cpu())
    x_test_scaled = scaler.transform(x_test.cpu())


    x_train_scaled = torch.from_numpy(x_train_scaled).to(device).float()

    x_test_scaled = torch.from_numpy(x_test_scaled).to(device).float()


    # import ipdb 
    # ipdb.set_trace()


    result_dict = defaultdict(list)
    for model_type in args.model_list:


        for trial in range(args.n_problems):
            result_dict["model"].append(model_type)

            # initialize model
            
            model, encode_time = init_model(model_type, args, x_train_scaled, y_train, x_test_scaled, y_test)

            result_dict["encode_time"].append(encode_time)


            model, train_time = train_model(model=model, features=x_train_scaled, labels=y_train,
                        num_epochs=args.num_epochs, lr=args.lr)

            result_dict["train_time"].append(train_time)

            # import ipdb 
            # ipdb.set_trace()
            preds, test_time = test_model(model=model, features=x_test_scaled)

            result_dict["test_time"].append(test_time)


            metrics = compute_metrics(y_true=y_test.long(), y_pred=preds)


            result_dict["acc"].append(metrics["acc"])
            result_dict["recall-micro"].append(metrics["recall-micro"])
            result_dict["recall-macro"].append(metrics["recall-macro"])
            result_dict["precision-micro"].append(metrics["precision-micro"])
            result_dict["precision-macro"].append(metrics["precision-macro"])
            result_dict["trial"].append(trial)
            print(model_type)
            print_metrics(metrics)

    


    model = RandomForestClassifier()
    rf_train_start_time = time.time()
    model.fit(x_train_scaled.cpu(), y_train.cpu())
    rf_train_end_time = time.time()

    rf_test_start_time = time.time()
    rf_preds = model.predict(x_test_scaled.cpu())
    rf_test_end_time = time.time()


    metrics = compute_metrics(y_true=torch.tensor(y_test).int(), y_pred=torch.tensor(rf_preds))


    print(f"RF")
    print(model)
    print_metrics(metrics)




    result_dict["model"].append("RF") 
    result_dict["encode_time"].append(-1)
    result_dict["train_time"].append(rf_train_end_time - rf_train_start_time)
    result_dict["test_time"].append(rf_test_end_time - rf_test_start_time)


    metrics = compute_metrics(y_true=y_test.long(), y_pred=rf_preds)


    result_dict["acc"].append(metrics["acc"])
    result_dict["recall-micro"].append(metrics["recall-micro"])
    result_dict["recall-macro"].append(metrics["recall-macro"])
    result_dict["precision-micro"].append(metrics["precision-micro"])
    result_dict["precision-macro"].append(metrics["precision-macro"])
    result_dict["trial"].append(0)





    model = DummyClassifier(strategy="uniform")
    model.fit(x_train_scaled.cpu(), y_train.cpu())


    random_preds = model.predict(x_test_scaled.cpu())

    metrics = compute_metrics(y_true=torch.tensor(y_test).int(), y_pred=torch.tensor(random_preds))


    print("RANDOM")
    print(model)
    print_metrics(metrics)




    result_dict["model"].append("RANDOM") 
    result_dict["encode_time"].append(-1)
    result_dict["train_time"].append(-1)
    result_dict["test_time"].append(-1)


    metrics = compute_metrics(y_true=y_test.long(), y_pred=random_preds)


    result_dict["acc"].append(metrics["acc"])
    result_dict["recall-micro"].append(metrics["recall-micro"])
    result_dict["recall-macro"].append(metrics["recall-macro"])
    result_dict["precision-micro"].append(metrics["precision-micro"])
    result_dict["precision-macro"].append(metrics["precision-macro"])

    result_dict["trial"].append(0)


    output_path = Path(f"{args.out_csv}")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(result_dict).to_csv(args.out_csv, index=False, sep='\t')




if __name__=='__main__':
    # Device configuration
    device = torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")
    args = get_args() 



    main()



from cProfile import run
import pickle    
import time

from sklearn.ensemble import RandomForestClassifier
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
# import multiprocessing as mp
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import deepchem as dc
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from pathlib import Path
import rdkit
from rdkit import Chem
from hdpy.ecfp_hd.encode_ecfp import ECFPEncoder
from hdpy.baseline_hd.classification_modules import RPEncoder
from hdpy.rff_hdc.encoder import RandomFourierEncoder
from hdpy.mole_hd.encode_smiles import SMILESHDEncoder, tokenize_smiles
from hdpy.ecfp_hd.encode_ecfp import compute_fingerprint_from_smiles
import random
random.seed(0)
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--smiles', nargs='*', default=['CC[N+](C)(C)Cc1ccccc1Br'])
parser.add_argument('--ngram-order', type=int, default=1)
parser.add_argument('--tokenizer', choices=["atomwise", "ngram"])
parser.add_argument('--D', type=int, help="size of encoding dimension", default=10000)
parser.add_argument('--dataset', choices=["bbbp", "sider", "clintox", "dude", "lit-pcba"], required=True)
parser.add_argument('--split-type', choices=["random", "scaffold"], required=True)
parser.add_argument('--input-feat-size', type=int, help="size of input feature dim. ", default=1024)
parser.add_argument('--model', choices=["smiles-pe", "ecfp", "rp", "rf"])
parser.add_argument('--n-trials', type=int, default=1, help="number of trials to perform")
parser.add_argument('--random-state', type=int, default=0)
args = parser.parse_args()



def train(model, hv_train, y_train, epochs=10):

    model.build_am(hv_train, y_train)

    for _ in range(epochs):

        model.retrain(hv_train, y_train)


def test(model, hv_test, y_test): 

    pred_list = model.predict(hv_test)

    # import ipdb 
    # ipdb.set_trace()
    eta_list = model.compute_confidence(hv_test)



    return {"y_pred": pred_list, "y_true": y_test, "eta": eta_list}



def functional_encode(datapoint):
    hv = torch.zeros(args.D)


    for pos, value in enumerate(datapoint):

        if isinstance(value,torch.Tensor):


            hv = hv + hd_model.item_mem["pos"][pos] * hd_model.item_mem["value"][value.data.int().item()]

        else:

            hv = hv + hd_model.item_mem["pos"][pos] * hd_model.item_mem["value"][value]

        # bind both item memory elements? or should I use a single 2 by n_bit matrix of values randomly chosen to associate with all possibilities?
        # hv = hv + (pos_hv * value_hv)

    # binarize
    hv = torch.where(hv > 0, hv, -1)
    hv = torch.where(hv <= 0, hv, 1)

    return hv


def collate_ecfp(batch):

    # print(batch)

    return torch.stack([functional_encode(x) for x in batch])



def run_hd_trial(smiles, labels, train_idxs, test_idxs):


    if args.model == "smiles-pe":

        hv_p, im_p =None, None


        #TODO: we need to figure out if we need to use a DUD-E target

        if args.dataset in ["dude", "lit-pcba"]:
            hv_p = Path(f".hd_cache/smiles-pe/{args.dataset}/{target_name}/{args.tokenizer}-{args.ngram_order}-dataset_hv.pth")
            im_p = Path(f".hd_cache/smiles-pe/{args.dataset}/{target_name}/{args.tokenizer}-{args.ngram_order}-item_mem.pth")

        else:
            hv_p = Path(f".hd_cache/smiles-pe/{args.dataset}/{args.tokenizer}-{args.ngram_order}-dataset_hv.pth")
            im_p = Path(f".hd_cache/smiles-pe/{args.dataset}/{args.tokenizer}-{args.ngram_order}-item_mem.pth")




        encode_time = 0
        dataset_hvs = None
        if not im_p.exists():
            toks = tokenize_smiles(smiles, tokenizer=args.tokenizer, ngram_order=args.ngram_order)
            im_p.parent.mkdir(parents=True, exist_ok=True)
            hd_model.build_item_memory(toks)
            encode_start = time.time()
            dataset_hvs = hd_model.encode_dataset(toks)
            encode_time = time.time() - encode_start
            torch.save(dataset_hvs, hv_p)
            torch.save(hd_model.item_mem, im_p)
        else:
            item_mem = torch.load(im_p)
            hd_model.item_mem = item_mem

            dataset_hvs = torch.load(hv_p)




    elif args.model == "ecfp":

        fps = None
        if args.dataset == "lit-pcba":
            if not (lit_pcba_path / Path("ecfp.npy")).exists():

                fps = [compute_fingerprint_from_smiles(x).reshape(1,-1) for x in tqdm(smiles, desc="computing fingerprints")]
                np.save(lit_pcba_path / Path("ecfp.npy"), fps)

            else:
                fps = np.load(lit_pcba_path/ Path("ecfp.npy"),allow_pickle=True)

        else:

                fps = [compute_fingerprint_from_smiles(x).reshape(1,-1) for x in tqdm(smiles, desc="computing fingerprints")]




        data = torch.from_numpy(np.concatenate(fps)).float()

        hv_p, im_p =None, None

        if args.dataset in ["dude", "lit-pcba"]:
            hv_p = Path(f".hd_cache/{args.model}/{args.dataset}/{target_name}/dataset_hv.pth")
            im_p = Path(f".hd_cache/{args.model}/{args.dataset}/{target_name}/item_mem.pth")

        else:
            hv_p = Path(f".hd_cache/{args.model}/{args.dataset}/dataset_hv.pth")
            im_p = Path(f".hd_cache/{args.model}/{args.dataset}/item_mem.pth")




        if not im_p.exists():
            im_p.parent.mkdir(parents=True, exist_ok=True)
            hd_model.build_item_memory(n_bits=data.shape[1])
            torch.save(hd_model.item_mem, im_p)
        else:
            item_mem = torch.load(im_p)
            hd_model.item_mem = item_mem


        encode_time = 0
        dataset_hvs = None
        if not hv_p.exists():
            encode_time = 0
            hv_p.parent.mkdir(parents=True, exist_ok=True)
            # with mp.Pool(mp.cpu_count() -1) as p:
            encode_start = time.time() # im putting this inside of the context manager to avoid overhead potentially
            # 

            dataloader = DataLoader(data, num_workers=70, batch_size=1000, collate_fn=collate_ecfp)
            dataset_hvs = []
            for batch_hvs in tqdm(dataloader, total=len(dataloader)):
                # batch_hvs = list(tqdm(p.imap(functional_encode, batch), total=len(batch), desc=f"encoding ECFPs with {mp.cpu_count() -1} workers.."))
                dataset_hvs.append(batch_hvs)
            dataset_hvs = torch.cat(dataset_hvs, dim=0)
            encode_time = time.time() - encode_start
            torch.save(dataset_hvs, hv_p)

        else:
            dataset_hvs = torch.load(hv_p)


       
    ##########################################################################################
    
    elif args.model == "rp":
        hd_model.cuda() 
        # todo: cache the actual model projection matrix too



        hv_p, im_p, dataset_fp_path =None, None, None

        if args.dataset in ["dude", "lit-pcba"]:
            hv_p = Path(f".hd_cache/{args.model}/{args.dataset}/{target_name}/dataset_hv.pth") 

            dataset_fp_path = Path(f".hd_cache/{args.model}/{args.dataset}/{target_name}/dataset_fps.pth")
        else:
            hv_p = Path(f".hd_cache/{args.model}/{args.dataset}/dataset_hv.pth")

            dataset_fp_path = Path(f".hd_cache/{args.model}/{args.dataset}/dataset_fps.pth")
        
        data = None
        if not dataset_fp_path.exists():
            dataset_fp_path.parent.mkdir(exist_ok=True, parents=True)
            fps = [compute_fingerprint_from_smiles(x).reshape(1,-1) for x in tqdm(smiles, desc="computing fingerprints")]
            fps = np.concatenate(fps)
            data = torch.from_numpy(fps).cuda()
            torch.save(data, dataset_fp_path)
        else:
            data = torch.load(dataset_fp_path)

        encode_time = 0
        dataset_hvs = None
        if not hv_p.exists():

            hv_p.parent.mkdir(parents=True, exist_ok=True)
            encode_start = time.time() # im putting this inside of the context manager to avoid overhead potentially
            dataloader = DataLoader(data, num_workers=0, batch_size=1000)
            dataset_hvs = []
            for batch_hvs in tqdm(dataloader, total=len(dataloader)):
                dataset_hvs.append(hd_model.encode(batch_hvs).cpu())

            dataset_hvs = torch.cat(dataset_hvs, dim=0)
            encode_time = time.time() - encode_start
            torch.save(dataset_hvs, hv_p)
        else:
            print(f"loading {args.model} precomputed hvs for {args.dataset}")
            dataset_hvs = torch.load(hv_p)

    
    hd_model.cuda()



    dataset_labels = torch.from_numpy(labels)


    dataset_hvs_train = torch.cat([dataset_hvs[idx].reshape(1,-1) for idx in train_idxs], dim=0)
    dataset_labels_train = torch.cat([dataset_labels[idx].reshape(1, -1) for idx in train_idxs], dim=0)

    dataset_hvs_test = torch.cat([dataset_hvs[idx].reshape(1,-1) for idx in test_idxs], axis=0)
    dataset_labels_test = torch.cat([dataset_labels[idx].reshape(1,-1) for idx in test_idxs], axis=0)

    train_start = time.time()
    train(hd_model, dataset_hvs_train, dataset_labels_train)
    train_time = time.time() - train_start 


    test_start = time.time()
    result_dict = test(hd_model, dataset_hvs_test, dataset_labels_test)
    test_time = time.time() - test_start

    # task_pred_array = np.array(result_dict["y_pred"][task_idx]).squeeze()
    result_dict["y_pred"] = result_dict["y_pred"].cpu().numpy()
    result_dict["eta"] = result_dict["eta"].cpu().numpy().reshape(-1,1)
    result_dict["y_true"] = result_dict["y_true"].cpu().numpy()
    result_dict["train_time"] = train_time
    result_dict["test_time"] = test_time
    result_dict["encode_time"] = encode_time
    result_dict["train_size"] = dataset_hvs_train.shape[0]
    result_dict["test_size"] = dataset_hvs_test.shape[0]



    result_dict["class_report"] = classification_report(y_pred=result_dict["y_pred"], y_true=result_dict["y_true"])

    try:
        result_dict["roc-auc"] = roc_auc_score(y_score=result_dict['eta'], y_true=dataset_labels_test.cpu().numpy())

    except ValueError as e:
        result_dict["roc-auc"] = None
        print(e)
    # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1


    print(result_dict["class_report"])
    print(f"roc-auc {result_dict['roc-auc']}")
    return result_dict



def run_sklearn_trial(smiles, labels, train_idxs, test_idxs):
    model = RandomForestClassifier() 

    dataset_fp_path = None

    if args.dataset in ["dude", "lit-pcba"]:
        dataset_fp_path = Path(f".hd_cache/{args.model}/{args.dataset}/{target_name}/dataset_fps.pth")
    else:
        dataset_fp_path = Path(f".hd_cache/{args.model}/{args.dataset}/dataset_fps.pth")
    
    data = None
    if not dataset_fp_path.exists():
        dataset_fp_path.parent.mkdir(exist_ok=True, parents=True)
        fps = [compute_fingerprint_from_smiles(x).reshape(1,-1) for x in tqdm(smiles, desc="computing fingerprints")]
        fps = np.concatenate(fps)
        data = torch.from_numpy(fps).cuda()
        torch.save(data, dataset_fp_path)
    else:
        data = torch.load(dataset_fp_path)

    encode_time = 0

    data = data.cpu()
    # dataset_labels = torch.from_numpy(labels)
    
    x_train, y_train = data[train_idxs], labels[train_idxs]
    x_test, y_test = data[test_idxs], labels[test_idxs]

    train_start = time.time()
    model.fit(x_train, y_train.ravel())
    train_time = time.time() - train_start 


    test_start = time.time()
    # result_dict = test(hd_model, dataset_hvs_test, dataset_labels_test)
    y_pred = model.predict(x_test)
    test_time = time.time() - test_start

    result_dict = {}
    # task_pred_array = np.array(result_dict["y_pred"][task_idx]).squeeze()
    result_dict["y_pred"] = y_pred
    result_dict["eta"] = model.predict_proba(x_test).reshape(-1,2)
    result_dict["y_true"] = y_test
    result_dict["train_time"] = train_time
    result_dict["test_time"] = test_time
    result_dict["encode_time"] = encode_time
    result_dict["train_size"] = x_train.shape[0]
    result_dict["test_size"] = x_test.shape[0]



    result_dict["class_report"] = classification_report(y_pred=result_dict["y_pred"], y_true=result_dict["y_true"])

    # import pdb 
    # pdb.set_trace()
    try:
        result_dict["roc-auc"] = roc_auc_score(y_score=result_dict['eta'][:, 1], y_true=y_test)

    except ValueError as e:
        result_dict["roc-auc"] = None
        print(e)
    # going from the MoleHD paper, we use their confidence definition that normalizes the distances between AM elements to between 0 and 1


    print(result_dict["class_report"])
    print(f"roc-auc {result_dict['roc-auc']}")
    return result_dict






def main(smiles, labels, train_idxs, test_idxs):
    trial_dict = {}
    for trial in range(args.n_trials):
        if args.model in ["smiles-pe", "ecfp", "rp"]:
            result_dict = run_hd_trial(smiles=smiles, labels=labels, train_idxs=train_idxs, test_idxs=test_idxs)
        else:
            result_dict = run_sklearn_trial(smiles=smiles, labels=labels, train_idxs=train_idxs, test_idxs=test_idxs)
        trial_dict[trial] = result_dict


    return trial_dict



if __name__ == "__main__":

    '''
        each problem (bbbp, sider, clintox) have n binary tasks..we'll form a separate AM for each
    '''

    if args.model == "smiles-pe":


        hd_model = SMILESHDEncoder(D=args.D)

    elif args.model == "ecfp":


        hd_model = ECFPEncoder(D=args.D)


    elif args.model == "rp":

        
        assert args.input_feat_size is not None
        assert args.D is not None
        hd_model = RPEncoder(input_size=args.input_feat_size, D=args.D, num_classes=2)

    # elif args.model == "rff":
# 
        # assert args.input_feat_size is not None
        # hd_model = RandomFourierEncoder(input_dim=args.input_feat_size, gamma=args.gamma, output_dim=args.D, n_feat_buckets=100)




    output_result_dir = Path("results")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True)




    print(args)



    if args.dataset == "bbbp":

        df = pd.read_csv('BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv')

        # probably want to use MOE_smiles
        smiles = df['rdkitSmiles'].values.tolist()
        labels = df['p_np'].values.reshape(-1,1)

        n_tasks = 1


        train_idxs, test_idxs = train_test_split(list(range(len(df))), random_state=args.random_state)

        result_dict = main(smiles, labels, train_idxs=train_idxs, test_idxs=test_idxs)

        with open(f"{output_result_dir}/bbbp_result.{args.model}.pkl", "wb") as handle:
            pickle.dump(result_dict, handle)

    elif args.dataset == "sider":

        df = pd.read_csv('/g/g13/jones289/workspace/hd-cuda-master/datasets/sider.csv')

        label_cols = [x for x in df.columns.values if "smiles" not in x] 
        smiles = df['smiles'].values.tolist()
        labels = df[[x for x in df.columns.values if "smiles" not in x]].values
        n_tasks = len(label_cols)



        # train_idxs, test_idxs = train_test_split(list(range(len(df))), random_state=args.random_state)


        train_idxs = None 
        test_idxs = None

        if args.split_type == "random":

            train_idxs, test_idxs = train_test_split(list(range(len(df))), random_state=args.random_state)
        elif args.split_type == "scaffold":

            scaffoldsplitter = dc.splits.ScaffoldSplitter()
            idxs = np.array(list(range(len(df))))

            dataset = dc.data.DiskDataset.from_numpy(X=idxs,w=np.zeros(len(df)),ids=df['smiles'])
            train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            train_idxs = train_data.X
            test_idxs = test_data.X


        for task_idx in range(n_tasks):

            result_dict = main(smiles, labels[:, task_idx], train_idxs=train_idxs, test_idxs=test_idxs)
            with open(f"{output_result_dir}/sider_task_{task_idx}-{args.model}-{args.split_type}.pkl", "wb") as handle:
                pickle.dump(result_dict, handle)


    elif args.dataset == "clintox":


        df = pd.read_csv("/g/g13/jones289/workspace/hd-cuda-master/datasets/clintox.csv")
        label_cols = [x for x in df.columns.values if "smiles" not in x]

        smiles = df['smiles'].values.tolist()
        labels = df[label_cols].values
        n_tasks = len(label_cols)

        train_idxs = None 
        test_idxs = None

        if args.split_type == "random":

            train_idxs, test_idxs = train_test_split(list(range(len(df))), random_state=args.random_state)
        elif args.split_type == "scaffold":


            scaffoldsplitter = dc.splits.ScaffoldSplitter()
            # idxs = np.array(list(range(len(df))))


            # had some issues with a few of these molecules so doing a filter for rdkit validity
            valid_list = [(idx, x) for idx, x in enumerate(df['smiles'].values.tolist()) if Chem.MolFromSmiles(x) is not None]

            print(f"dropped {len(df) - len(valid_list)} invalid smiles for scaffold splitting")

            dataset = dc.data.DiskDataset.from_numpy(X=[x[0] for x in valid_list],w=np.zeros(len(valid_list)),ids=[x[1] for x in valid_list])


            train_data, test_data = scaffoldsplitter.train_test_split(dataset)

            train_idxs = train_data.X
            test_idxs = test_data.X

        for task_idx in range(n_tasks):
            result_dict = main(smiles, labels[:, task_idx], train_idxs=train_idxs, test_idxs=test_idxs)
            with open(f"{output_result_dir}/clintox_task_{task_idx}-{args.model}-{args.split_type}.pkl", "wb") as handle:

                pickle.dump(result_dict, handle)


    elif args.dataset == "dude":

        dude_data_p = Path("/usr/workspace/atom/gbsa_modeling/dude_smiles/")
        for dude_smiles_path in dude_data_p.glob("*_gbsa_smiles.csv"):

            target_name = dude_smiles_path.name.split("_")[0]
            smiles_df = pd.read_csv(dude_smiles_path)

            # for split_type in ["random", "scaffold"]:

            for split_type in ["random"]:

                dude_split_path = None
                if split_type == "random":
                    dude_split_path = dude_smiles_path.with_name(dude_smiles_path.stem+"_with_base_rdkit_smiles_train_valid_test_random_random.csv")
                # else:
                    # dude_split_path = dude_smiles_path.with_name(dude_smiles_path.stem+"_with_base_rdkit_smiles_train_valid_test_scaffold_scaffold.csv")

                split_df = pd.read_csv(dude_split_path)

                df = pd.merge(smiles_df, split_df, left_on="id", right_on="cmpd_id")

                train_idxs, test_idxs = [], []

                for grp_name, grp_df in df.groupby('subset'):
                    if grp_name == "train":
                        train_idxs = grp_df.index.values.tolist()
                    elif grp_name == "test":
                        test_idxs = grp_df.index.values.tolist()

                smiles = df['smiles'].values.tolist()
                labels = df['decoy'].apply(lambda x: int(not x)).values.reshape(-1,1)

                n_tasks = 1

                result_dict = main(smiles, labels, train_idxs=train_idxs, test_idxs=test_idxs)

                # import ipdb 
                # ipdb.set_trace()

                with open(f"{output_result_dir}/dude_{target_name}_{split_type}.{args.model}.pkl", "wb") as handle:
                    pickle.dump(result_dict, handle)

    elif args.dataset == "lit-pcba":

        # lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/")
        # lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/ADRB2/actives/actives/ecfp/data.npy")

        # import ipdb
        # ipdb.set_trace()
        lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/")
        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name


            print(lit_pcba_path)



            # '''
            actives_df = pd.read_csv(list(lit_pcba_path.glob("actives.smi"))[0], header=None, delim_whitespace=True)
            # actives_df['subset'] = ['train'] * len(actives_df)
            actives_df['label'] = [1] * len(actives_df)


            inactives_df = pd.read_csv(list(lit_pcba_path.glob("inactives.smi"))[0], header=None, delim_whitespace=True)
            # inactives_df['subset'] = ['train'] * len(inactives_df)
            inactives_df['label'] = [0] * len(inactives_df)

            df = pd.concat([actives_df, inactives_df]).reset_index(drop=True)
            # '''

           
            train_idxs, test_idxs = train_test_split(list(range(len(df))), stratify=df["label"], random_state=args.random_state)

            # train_idxs, test_idxs = [], []

            # for grp_name, grp_df in df.groupby('subset'):
                # if grp_name == "train":
                    # train_idxs = grp_df.index.values.tolist()
                # elif grp_name == "test":
                    # test_idxs = grp_df.index.values.tolist() 

            smiles = df[0].values.tolist()
            labels = df['label'].values.reshape(-1,1)

            n_tasks = 1

            result_dict = main(smiles, labels, train_idxs=train_idxs, test_idxs=test_idxs)

            # import ipdb 
            # ipdb.set_trace()

            with open(f"{output_result_dir}/lit_pcba_{target_name}_random_strat_split.{args.model}.pkl", "wb") as handle:
                pickle.dump(result_dict, handle)


            # '''

    '''
    elif args.dataset == "lit-pcba-ave":

        # lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/")
        # lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/ADRB2/actives/actives/ecfp/data.npy")
        lit_pcba_data_p = Path("/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/")
        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name


            print(lit_pcba_path)



            actives_train_df = pd.read_csv(list(lit_pcba_path.glob("*_active_T.smi"))[0], header=None, delim_whitespace=True)
            actives_train_df['subset'] = ['train'] * len(actives_train_df)
            actives_train_df['label'] = [1] * len(actives_train_df)


            inactives_train_df = pd.read_csv(list(lit_pcba_path.glob("*_inactive_T.smi"))[0], header=None, delim_whitespace=True)
            inactives_train_df['subset'] = ['train'] * len(inactives_train_df)
            inactives_train_df['label'] = [0] * len(inactives_train_df)

            actives_test_df = pd.read_csv(list(lit_pcba_path.glob("*_active_V.smi"))[0], header=None, delim_whitespace=True)
            actives_test_df['subset'] = ['test'] * len(actives_test_df)
            actives_test_df['label'] = [1] * len(actives_test_df)

            inactives_test_df = pd.read_csv(list(lit_pcba_path.glob("*_inactive_V.smi"))[0], header=None, delim_whitespace=True)
            inactives_test_df['subset'] = ['test'] * len(inactives_test_df)
            inactives_test_df['label'] = [0] * len(inactives_test_df)

            df = pd.concat([actives_train_df, inactives_train_df, actives_test_df, inactives_test_df]).reset_index(drop=True)



            target_name = dude_smiles_path.name.split("_")[0]
            smiles_df = pd.read_csv(dude_smiles_path)

            # for split_type in ["random", "scaffold"]:

            for split_type in ["ave"]:

                dude_split_path = None
                # if split_type == "random":
                    # dude_split_path = dude_smiles_path.with_name(dude_smiles_path.stem+"_with_base_rdkit_smiles_train_valid_test_random_random.csv")
                # else:
                    # dude_split_path = dude_smiles_path.with_name(dude_smiles_path.stem+"_with_base_rdkit_smiles_train_valid_test_scaffold_scaffold.csv")

                split_df = pd.read_csv(dude_split_path)

                df = pd.merge(smiles_df, split_df, left_on="id", right_on="cmpd_id")

            

            train_idxs, test_idxs = [], []

            for grp_name, grp_df in df.groupby('subset'):
                if grp_name == "train":
                    train_idxs = grp_df.index.values.tolist()
                elif grp_name == "test":
                    test_idxs = grp_df.index.values.tolist() 

            smiles = df[0].values.tolist()
            labels = df['label'].values.reshape(-1,1)

            n_tasks = 1

            result_dict = main(smiles, labels, train_idxs=train_idxs, test_idxs=test_idxs)

            # import ipdb 
            # ipdb.set_trace()

            with open(f"{output_result_dir}/lit_pcba_{target_name}_ave_split.{args.model}.pkl", "wb") as handle:
                pickle.dump(result_dict, handle)


    '''
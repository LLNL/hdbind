from pathlib import Path
import time     
import pandas as pd
from tqdm import tqdm 
import torch
import torch.multiprocessing as mp
from hdpy.hd_model import HDModel
from tqdm import tqdm
import pickle 
# from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
# import multiprocessing as mp
# import functools
import numpy as np
from hdpy.ecfp_hd.encode_ecfp import ECFPEncoder
import mdtraj
import pandas as pd
#todo: contact map https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/ 


class PROTHDEncoder(HDModel):

    def __init__(self, D):
        super(PROTHDEncoder, self).__init__()

        # "D" is the dimension of the encoded representation
        self.D = D

    def build_item_memory(self, dataset_tokens):
        self.item_mem = {}

        if not isinstance(dataset_tokens[0], list):
            dataset_tokens = [dataset_tokens]

        print("building item memory")
        for tokens in tqdm(dataset_tokens):

            tokens = list(set(tokens))
            # "empty" token?
            for token in tokens:
                if token not in self.item_mem.keys():
                    # print(token)
                    # draw a random vector from 0->1, convert to binary (i.e. if < .5), convert to polarized
                    token_hv = torch.bernoulli(torch.empty(self.D).uniform_(0,1))
                    token_hv = torch.where(token_hv > 0 , token_hv, -1).int() 
                    self.item_mem[token] = token_hv

        print(f"item memory formed with {len(self.item_mem.keys())} entries.")

    def encode(self, tokens):

        # tokens is a list of tokens, i.e. it corresponds to 1 sample

        hv = torch.zeros(self.D).int()

        for idx, token in enumerate(tokens):
            token_hv = self.item_mem[token]
            hv = hv + torch.roll(token_hv, idx).int()


        # binarize
        hv = torch.where(hv > 0, hv, -1).int()
        hv = torch.where(hv <= 0, hv, 1).int()
        return hv


class PDBBindHD(HDModel):

    def featurize(  # type: ignore[override]
            self, protein_file: str, ligand_file:str, ligand_encoder:ECFPEncoder) -> np.ndarray:


        residue_map = {
        "ALA": 0, 
        "ARG": 0,
        "ASN": 0,
        "ASP": 0,
        "CYS": 0, 
        "GLN": 0, 
        "GLU": 0, 
        "GLY": 0, 
        "HIS": 0, 
        "ILE": 0,
        "LEU": 0, 
        "LYS": 0, 
        "MET": 0, 
        "PHE": 0, 
        "PRO": 0, 
        "PYL": 0, 
        "SER": 0, 
        "SEC": 0, 
        "THR": 0, 
        "TRP": 0,
        "TYR": 0, 
        "VAL": 0, 
        "ASX": 0, 
        "GLX": 0
    }

        protein = mdtraj.load(protein_file)

        for atom in protein.top.atoms:

            if atom.residue.name not in residue_map.keys():
                pass
            else:
                residue_map[atom.residue.name] += 1

        prot_vec = torch.from_numpy(np.array(list(residue_map.values())).reshape(1, -1))

        from hdpy.baseline_hd.classification_modules import RPEncoder
        rp_encoder = RPEncoder(D=10000, input_size=24, num_classes=2).cpu()
        prot_hv = rp_encoder.encode(prot_vec)

        # this should be constructed outside of this 

        # import ipdb 
        # ipdb.set_trace()
        from rdkit import Chem
        # mol_supp = Chem.SDMolSupplier(str(ligand_file))
        # mol = next(mol_supp)

        # see this for more information https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind/
        mol = Chem.MolFromMol2File(str(ligand_file))

        smiles = Chem.MolToSmiles(mol)

        from hdpy.ecfp_hd.encode_ecfp import compute_fingerprint_from_smiles
        # smiles = pd.read_csv(ligand_file, sep="\t", header=None)[0].values[0]


        fp  = compute_fingerprint_from_smiles(smiles)
        lig_hv = lig_encoder.encode(fp)

        # import ipdb
        # ipdb.set_trace()
        complex_hv = prot_hv * lig_hv

        return complex_hv


def job(pdbid_tup:tuple):

    pdbid, pdbid_df = pdbid_tup
    pocket_f = Path(f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_pocket.pdb")
    ligand_f = Path(f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_ligand.mol2")

    try:
        data = featurizer.featurize(protein_file=pocket_f, 
                            ligand_file=ligand_f,
                            ligand_encoder=lig_encoder)
    
        affinity = pdbid_df["-logKd/Ki"].values[:]
        label = affinity > 6 or affinity < 4
        return (data, label)

    except Exception as e:
        print(e)
        return


def train(model, hv_train, y_train, epochs=10):

    single_pass_train_start = time.time()
    model.build_am(hv_train, y_train)
    single_pass_train_time = time.time() - single_pass_train_start


    learning_curve_list = []

    retrain_start = time.time()
    for _ in range(epochs):

        mistake_ct = model.retrain(hv_train, y_train, return_mistake_count=True)
        learning_curve_list.append(mistake_ct)

    retrain_time = time.time() - retrain_start
    return learning_curve_list, single_pass_train_time, retrain_time



def test(model, hv_test, y_test):

    # import pdb
    # pdb.set_trace()
    test_start = time.time()
    pred_list = model.predict(hv_test)
    test_time = time.time() - test_start

    conf_test_start = time.time()
    eta_list = model.compute_confidence(hv_test)
    conf_test_time = time.time() - conf_test_start


    return {"y_pred": pred_list, "y_true": y_test, "eta": eta_list, "test_time": test_time, "conf_test_time": conf_test_time}



if __name__ == "__main__":

    import sys

    p = float(sys.argv[1])

    df = pd.read_csv("/g/g13/jones289/workspace/fast_md/data/metadata/pdbbind_2016_train_val_test.csv").sample(frac=p)

    #todo: switch v2016 to updated pdbbind version
    result_path = Path("prototype_data.pkl")


    split_dict = {"general_train": {"data_list": [], "label_list": [], "pdbid_list": []},
                  "general_val": {"data_list": [], "label_list": [], "pdbid_list": []},
                  "refined_train": {"data_list": [], "label_list": [], "pdbid_list": []},
                  "refined_val":{"data_list": [], "label_list": [],  "pdbid_list": []},
                  "core_test":{"data_list": [], "label_list": [], "pdbid_list": []}}

    if not result_path.exists():

        featurizer = PDBBindHD()
        lig_encoder = ECFPEncoder(D=10000)
        lig_encoder.build_item_memory(n_bits=1024)

        # data_list =[]
        # label_list = []

        for pdbid, pdbid_df in tqdm(df.groupby("name")):


            split = pdbid_df["set"].to_numpy()[0]

            pocket_f = Path(f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_pocket.pdb")
            ligand_f = Path(f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_ligand.mol2")

            try:
                data = featurizer.featurize(protein_file=pocket_f, 
                                    ligand_file=ligand_f,
                                    ligand_encoder=lig_encoder)
            
                affinity = pdbid_df["-logKd/Ki"].values[:]
                label = affinity > 6 or affinity < 4

                split_dict[split]["data_list"].append(data)
                split_dict[split]["label_list"].append(label)
                split_dict[split]["pdbid_list"].append(pdbid)

            except Exception as e:
                print(e)

        out_dict = {"split_dict": split_dict,
                    "featurizer": featurizer,
                    "lig_encoder": lig_encoder}
        
        with open(result_path, "wb") as handle:
            pickle.dump(out_dict, handle)

    else:
        with open(result_path, "rb") as handle:
            out_dict = pickle.load(handle)



    import ipdb 
    ipdb.set_trace()

    train_hv_list = [value["data_list"] for key, value in out_dict["split_dict"].items() if key in ["general_train", "refined_train"]]
    train_label_list = [value["label_list"] for key, value in out_dict["split_dict"].items() if key in ["general_train", "refined_train"]]

    test_hv_list = [value["data_list"] for key, value in out_dict["split_dict"].items() if key in ["core_test"]]
    test_label_list = [value["label_list"] for key, value in out_dict["split_dict"].items() if key in ["core_test"]]
    # from hdpy.hd_main import train, test 

    '''
    hvs = torch.cat(out_dict["data"])
    labels = torch.from_numpy(np.concatenate(out_dict["label"])).int().reshape(-1,1)
    pdbid_list = np.array(out_dict["pdbid"])

    # I'm using exisitng splits to group the pdbids together in to train/test splits
    train_list = []
    test_list = []
    for grp_name, grp_df in df.groupby("set"):
        if grp_name in ["general_train", "refined_train"]:
            train_list.append(grp_df)
        elif grp_name in ["core_test"]:
            test_list.append(grp_df)

    # concatenate the dataframes which contain the pdbids
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)


    train_hv_list = []
    test_hv_list = []
    train_label_list = []
    test_label_list = []

    #collect the train hypervectors and labels
    for pdbid in train_df["name"]:

        idx = np.where(pdbid_list == pdbid)[0]

        if idx.shape[0] == 0:
            pass
        # if pdbid == "3mf5":
            # import ipdb 
            # ipdb.set_trace()
        else:
            print(idx, len(train_df), pdbid)
            train_hv_list.append(hvs[idx])
            train_label_list.append(labels[idx])


    # collect the test hypervectors and labels 
    for pdbid in test_df["name"]:
        idx = np.where(pdbid_list == pdbid)[0]
        if idx.shape[0] == 0:
            pass
        else:
            print(idx, len(test_df), pdbid)
            test_hv_list.append(hvs[idx])
            test_label_list.append(labels[idx])

    '''
    # train_hvs = torch.cat(train_hv_list)
    # test_hvs = torch.cat(test_hv_list)
    train_hvs = torch.cat([torch.cat(x) for x in train_hv_list])
    test_hvs = torch.cat([torch.cat(x) for x in test_hv_list])


    train_labels = torch.from_numpy(np.concatenate(train_label_list)).int()
    test_labels = torch.from_numpy(np.concatenate(test_label_list)).int()
    # train_labels = torch.cat(train_label_list)
    # test_labels = torch.cat(test_label_list)

    # fit the model on the training set


    print(train_hvs.shape, test_hvs.shape, train_labels.shape, test_labels.shape)
    learning_curve_list, single_pass_train_time, retrain_time = train(model=out_dict["featurizer"], hv_train=train_hvs, y_train=train_labels,epochs=100)

    result_dict = test(model=out_dict["featurizer"], hv_test=test_hvs, y_test=test_labels)

    with open("result_dict.pkl", "wb") as handle:
        pickle.dump(result_dict, handle)

    from sklearn.metrics import classification_report, roc_auc_score

    print(f"roc_auc: {roc_auc_score(y_true=result_dict['y_true'], y_score=result_dict['eta'])}")
    print(classification_report(y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]))
    '''
    # print(pdbid, activity)

    # import multiprocessing as mp
    # mp.set_start_method("spawn")

    # with mp.Pool(64) as p:

        # result = list(tqdm(p.imap(job, list(df.groupby("name"))), total=len(df)))


    # data = torch.cat([x[0] for x in result])
    # labels = torch.cat(x[1] for x in result)
    '''
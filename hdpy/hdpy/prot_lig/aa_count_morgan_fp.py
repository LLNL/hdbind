from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
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
from rdkit import Chem
from hdpy.ecfp_hd.encode_ecfp import compute_fingerprint_from_smiles
import ipdb 
# ipdb.set_trace()
#todo: contact map https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/ 
from hdpy.baseline_hd.classification_modules import RPEncoder
from scipy.spatial import distance
# from graphein.protein.config import ProteinGraphConfig
# from graphein.protein.graphs import construct_graph
# from graphein.molecule import MoleculeGraphConfig
# import graphein.molecule as gm


from openbabel import pybel 
from tf_bio_data import featurize_pybel_complex
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph



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
    def __init__(self, D:int):
        super(HDModel, self).__init__()

        self.D = D
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

        rp_encoder = RPEncoder(D=10000, input_size=24, num_classes=2).cpu()
        prot_hv = rp_encoder.encode(prot_vec)

        # this should be constructed outside of this 


        # see this for more information https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind/
        mol = Chem.MolFromMol2File(str(ligand_file))

        smiles = Chem.MolToSmiles(mol)
        fp  = compute_fingerprint_from_smiles(smiles)
        lig_hv = lig_encoder.encode(fp)

        complex_hv = prot_hv * lig_hv

        return complex_hv


class ComplexGraphHD(HDModel):

    def __init__(self, D:int, node_feat_size:int):
        super(HDModel, self).__init__()
        self.D = D
        self.node_feat_size = node_feat_size
        self.rp_encoder = RPEncoder(input_size=self.node_feat_size, D=self.D, num_classes=2)

    def featurize(self, protein_file:Path, ligand_file:Path, ligand_encoder:None):

        # import ipdb
        # ipdb.set_trace()
        # prot_config = ProteinGraphConfig()
        # lig_config = MoleculeGraphConfig()
        # g = construct_graph(config, pdb_code=protein_file.parent.name)

        # p_g = construct_graph(prot_config, path=str(protein_file))
        # l_g = gm.construct_graph(lig_config, path=str(ligand_file))


        ligand_mol = next(pybel.readfile("mol2", str(ligand_file.with_suffix(".mol2"))))
        pocket_mol = next(pybel.readfile("mol2", str(protein_file.with_suffix(".mol2"))))


        data = torch.from_numpy(featurize_pybel_complex(ligand_mol=ligand_mol, pocket_mol=pocket_mol)).float()

        #map node features to binary (bipolar) using random projection
        node_hvs = self.rp_encoder.encode(data)

        # compute pairwise distances
        pdists = distance.squareform(distance.pdist(data[:, :3]) <= 1.5)
        pdists = torch.from_numpy(pdists)
        edge_index, _ = dense_to_sparse(pdists)

        graph_data = Data(x=data, node_hvs=node_hvs, edge_index=edge_index, num_nodes=data.shape[0])        

        # convert to pytorch geometric object, use the torch_geometric.utils.k_hop_subgraph function

        import ipdb
        ipdb.set_trace()

        subgraph_1_hop_list = []
        subgraph_2_hop_list = []
        for node_idx in range(graph_data.num_nodes):

            node_subset_1_hop, _, _, edge_mask_1_hop = k_hop_subgraph([node_idx], num_hops=1, edge_index=edge_index, relabel_nodes=False)
            node_subset_2_hop, _, _, edge_mask_2_hop = k_hop_subgraph([node_idx], num_hops=2, edge_index=edge_index, relabel_nodes=False)


            hv_1_hop = graph_data.node_hvs[node_subset_1_hop].sum(dim=0)
            hv_2_hop = graph_data.node_hvs[node_subset_2_hop].sum(dim=0)



            # why just two hops? we can specify this as a parameter

            subgraph_1_hop_list.append(edge_mask_1_hop.unsqueeze(dim=0))
            subgraph_2_hop_list.append(edge_mask_2_hop.unsqueeze(dim=0))

            # ipdb.set_trace()

        graph_data.subgraph_1_hop_mask = torch.cat(subgraph_1_hop_list)
        graph_data.subgraph_2_hop_mask = torch.cat(subgraph_2_hop_list)

        # create node feature hypervectors

        # p_hv = torch.empty(1, self.D).uniform_(0, 1)
        # p_hv = (2*torch.bernoulli(p_hv)) - 1

        # find torch function that rotates hypervector, use this to assign a value to each atom in the graph to form the node feature matrix

        import ipdb
        ipdb.set_trace()




        #relation embedding stage 

        for node in range(graph_data.num_nodes):

            hv_1_hop = graph_data.subgraph_1_hop_mask[node].sum(dim=0)
            hv_2_hop = graph_data.subgraph_2_hop_mask[node].sum(dim=0)


        

        return graph_data 






































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


        # here is a good place to use command line args to choose the featurizer for the complex
        # todo (derek): does it make sense to build the ligand item memory here?
        # featurizer = PDBBindHD(D=10000)
        featurizer = ComplexGraphHD(D=10000, node_feat_size=22)
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

                label = None
                if affinity > 6:
                    label = 1
                elif affinity < 4:
                    label = 0 
                else:
                    # this would be ambiguous so we toss these examples
                    print("ambiguous binder, skipping")
                    continue

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

    train_hv_list = [value["data_list"] for key, value in out_dict["split_dict"].items() if key in ["general_train", "refined_train"]]
    train_label_list = [value["label_list"] for key, value in out_dict["split_dict"].items() if key in ["general_train", "refined_train"]]

    test_hv_list = [value["data_list"] for key, value in out_dict["split_dict"].items() if key in ["core_test"]]
    test_label_list = [value["label_list"] for key, value in out_dict["split_dict"].items() if key in ["core_test"]]

    train_hvs = torch.cat([torch.cat(x) for x in train_hv_list])
    test_hvs = torch.cat([torch.cat(x) for x in test_hv_list])


    train_labels = torch.from_numpy(np.concatenate(train_label_list)).int()
    test_labels = torch.from_numpy(np.concatenate(test_label_list)).int()

    # fit the model on the training set


    print(train_hvs.shape, test_hvs.shape, train_labels.shape, test_labels.shape)
    learning_curve_list, single_pass_train_time, retrain_time = train(model=out_dict["featurizer"], hv_train=train_hvs, y_train=train_labels,epochs=10)

    result_dict = test(model=out_dict["featurizer"], hv_test=test_hvs, y_test=test_labels)
    result_dict["featurizer"] = out_dict["featurizer"]
    result_dict["lig_encoder"] = out_dict["lig_encoder"]

    print(f"roc_auc: {roc_auc_score(y_true=result_dict['y_true'], y_score=result_dict['eta'])}")
    print(classification_report(y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]))

    from sklearn.dummy import DummyClassifier

    uni_dummy_clf = DummyClassifier(strategy="uniform")
    uni_dummy_clf.fit(X=train_labels.numpy(), y=train_labels.numpy())
    uni_dummy_probs = uni_dummy_clf.predict_proba(X=test_labels.numpy())
    uni_dummy_roc_auc = roc_auc_score(y_true=test_labels.numpy(), y_score=uni_dummy_probs[:, 1])
    print(f"roc_auc (random): {uni_dummy_roc_auc}")


    most_freq_clf = DummyClassifier(strategy="most_frequent")
    most_freq_clf.fit(X=train_labels.numpy(), y=train_labels.numpy())
    most_freq_probs = most_freq_clf.predict_proba(X=test_labels.numpy())
    most_freq_roc_auc = roc_auc_score(y_true=test_labels.numpy(), y_score=most_freq_probs[:, 1])
    print(f"roc_auc (most frequent): {most_freq_roc_auc}")


    # import ipdb 
    # ipdb.set_trace()
    with open("result_dict.pkl", "wb") as handle:
        pickle.dump(result_dict, handle)
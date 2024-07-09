################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
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
from rdkit.rdBase import BlockLogs
from openbabel import openbabel as ob

ob_log_handler = ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
from torch.utils.data import ConcatDataset

# from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
# import multiprocessing as mp
# import functools
import numpy as np
from hdpy.hdpy.ecfp_hd.encode import StreamingECFPEncoder
import mdtraj
import pandas as pd
from rdkit import Chem
from hdpy.hdpy.ecfp_hd.encode import compute_fingerprint_from_smiles
from torch.utils.data import Dataset
import ipdb
from torch_geometric.loader import DataLoader

# ipdb.set_trace()
# todo: contact map https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/protein_contact_map/
from hdpy.baseline_hd.classification_modules import RPEncoder
from scipy.spatial import distance


from openbabel import pybel
from tf_bio_data import featurize_pybel_complex
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph

"""
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
                    token_hv = torch.bernoulli(torch.empty(self.D).uniform_(0, 1))
                    token_hv = torch.where(token_hv > 0, token_hv, -1).int()
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
"""
"""
# class DisjointComplexEncoder(HDModel):
    def __init__(self, D: int):
        # super(HDModel, self).__init__()
        super().__init__()

        self.D = D
        self.ligand_encoder = StreamingECFPEncoder(D=self.D)
        self.protein_encoder = RPEncoder(D=self.D, input_size=24, num_classes=2) #why is number of classes an argument?

    def featurize(  # type: ignore[override]
        self, protein_file: str, ligand_file: str, ligand_encoder: StreamingECFPEncoder
    ) -> np.ndarray:

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
            "GLX": 0,
        }

        protein = mdtraj.load(protein_file)

        for atom in protein.top.atoms:

            if atom.residue.name not in residue_map.keys():
                pass
            else:
                residue_map[atom.residue.name] += 1

        prot_vec = torch.from_numpy(np.array(list(residue_map.values())).reshape(1, -1))

        prot_hv = self.protein_encoder.encode(prot_vec)

        # this should be constructed outside of this

        # see this for more information https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind/
        mol = Chem.MolFromMol2File(str(ligand_file))

        smiles = Chem.MolToSmiles(mol)
        fp = compute_fingerprint_from_smiles(smiles)
        lig_hv = self.lig_encoder.encode(fp)

        complex_hv = prot_hv * lig_hv

        return complex_hv
"""
# class GraphData(Data):

# def __init__(self):
# super(GraphData).__init__(self)
# def __cat_dim__(self, key, value, *args, **kwargs):
# if key == 'foo':
# return None
# return super().__cat_dim__(None, value, *args, **kwargs)


class DisjointComplexDataset(Dataset):
    def __init__(self, data_dir: Path, meta_path: Path, D: int, p: float, split: str):
        # super(DisjointComplexDataset, self).__init__()
        super().__init__()
        self.data_dir = data_dir
        self.D = D
        self.p = p
        self.split = split

        self.pdbid_list = []
        self.data_dict = {}

        self.data_cache_dir = Path(
            "/p/lustre2/jones289/data/hdc_pdbbind/disjoint_complex_hd_cache/"
        )

        if not self.data_cache_dir.exists():
            self.data_cache_dir.mkdir(parents=True, exist_ok=True)

        self.meta_df = pd.read_csv(meta_path).groupby("set").sample(frac=p)

        self.meta_df = self.meta_df[self.meta_df["set"] == split]

        # label_list = []
        self.label_dict = {}

        for pdbid, _ in self.meta_df.groupby("pdbid"):
            affinity = self.meta_df.loc[self.meta_df["pdbid"] == pdbid][
                "-logKd/Ki"
            ].values
            label = None
            # affinity = self.label_dict[pdbid]
            if affinity >= 6:
                label = 1
            elif affinity <= 4:
                label = 0
            else:
                # this would be ambiguous so we toss these examples
                print("ambiguous binder, skipping")
                continue
            self.label_dict[pdbid] = torch.tensor(label).int()

        for pdbid_path in self.data_dir.glob("*/"):
            pdbid = pdbid_path.name
            if len(pdbid_path.name) == 4:
                if pdbid in self.label_dict.keys():
                    self.pdbid_list.append(pdbid_path.name)

        self.protein_encoder = RPEncoder(input_size=24, D=self.D, num_classes=2)

        # import ipdb
        # ipdb.set_trace()
        self.ligand_encoder = StreamingECFPEncoder(D=self.D)
        self.ligand_encoder.build_item_memory(n_bits=1024)

    # def featurize(  # type: ignore[override]
    # self, protein_file: str, ligand_file: str, ligand_encoder: StreamingECFPEncoder
    # ) -> np.ndarray:
    def featurize(self, pdbid: str) -> np.ndarray:  # type: ignore[override]
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
            "GLX": 0,
        }

        # protein_file = self.data_dir / Path(f"{pdbid}/{pdbid}_pocket.mol2")
        protein_file = self.data_dir / Path(f"{pdbid}/{pdbid}_pocket.pdb")
        ligand_file = self.data_dir / Path(f"{pdbid}/{pdbid}_ligand.mol2")

        # print(protein_file, ligand_file)

        protein = mdtraj.load(protein_file)

        for atom in protein.top.atoms:
            if atom.residue.name not in residue_map.keys():
                pass
            else:
                residue_map[atom.residue.name] += 1

        prot_vec = torch.from_numpy(np.array(list(residue_map.values())).reshape(1, -1))

        prot_hv = self.protein_encoder.encode(prot_vec)

        # this should be constructed outside of this

        try:
            # see this for more information https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind/
            # mol = Chem.MolFromMol2File(str(ligand_file), sanitize=False, removeHs=False, cleanupSubstructures=False)
            mol = Chem.MolFromMol2File(str(ligand_file), sanitize=False)
            # import ipdb
            # ipdb.set_trace()

            smiles = Chem.MolToSmiles(mol)
            fp = compute_fingerprint_from_smiles(smiles)
            lig_hv = self.ligand_encoder.encode(fp)

            complex_hv = prot_hv * lig_hv

            return complex_hv
        except Exception as e:
            print(e)
            return prot_hv

    def __getitem__(self, idx):
        pdbid = self.pdbid_list[idx]

        # if pdbid not in self.data_dict.keys():
        graph_data = self.featurize(pdbid)

        return graph_data, self.label_dict[pdbid]
        # return self.data_dict[pdbid]

    def __len__(self):
        return len(self.pdbid_list)


class ComplexGraphHDDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        D: int,
        node_feat_size: int,
        meta_path: Path,
        p: float,
        split: str,
    ):
        super(ComplexGraphHDDataset, self).__init__()
        self.data_dir = data_dir
        self.D = D
        self.node_feat_size = node_feat_size
        self.meta_path = meta_path
        self.rp_encoder = RPEncoder(
            input_size=self.node_feat_size, D=self.D, num_classes=2
        )
        self.split = split

        self.pdbid_list = []
        self.data_dict = {}

        self.data_cache_dir = Path(
            "/p/lustre2/jones289/data/hdc_pdbbind/complex_graph_hd_cache/"
        )

        if not self.data_cache_dir.exists():
            self.data_cache_dir.mkdir(parents=True, exist_ok=True)

        self.meta_df = pd.read_csv(meta_path).groupby("set").sample(frac=p)

        self.meta_df = self.meta_df[self.meta_df["set"] == split]

        self.label_dict = {}

        for pdbid, _ in self.meta_df.groupby("pdbid"):
            affinity = self.meta_df.loc[self.meta_df["pdbid"] == pdbid][
                "-logKd/Ki"
            ].values
            label = None
            if affinity >= 6:
                label = 1
            elif affinity <= 4:
                label = 0
            else:
                # this would be ambiguous so we toss these examples
                print("ambiguous binder, skipping")
                continue
            self.label_dict[pdbid] = torch.tensor(label)

        for pdbid_path in self.data_dir.glob("*/"):
            pdbid = pdbid_path.name
            if len(pdbid_path.name) == 4:
                if pdbid in self.label_dict.keys():
                    self.pdbid_list.append(pdbid_path.name)

    def __getitem__(self, idx):
        pdbid = self.pdbid_list[idx]

        graph_data = self.featurize(pdbid)

        return graph_data

    def __len__(self):
        return len(self.pdbid_list)

    def featurize(self, pdbid):
        protein_file = self.data_dir / Path(f"{pdbid}/{pdbid}_pocket.pdb")
        ligand_file = self.data_dir / Path(f"{pdbid}/{pdbid}_ligand.mol2")

        cache_file = Path(self.data_cache_dir) / Path(f"{pdbid}.pt")

        # if cache_file.exists():
        # graph_data = torch.load(cache_file)
        # return graph_data

        # else:

        ligand_mol = next(pybel.readfile("mol2", str(ligand_file.with_suffix(".mol2"))))
        pocket_mol = next(
            pybel.readfile("mol2", str(protein_file.with_suffix(".mol2")))
        )

        data = torch.from_numpy(
            featurize_pybel_complex(ligand_mol=ligand_mol, pocket_mol=pocket_mol)
        ).float()

        # map node features to binary (bipolar) using random projection
        node_hvs = self.rp_encoder.encode(data)

        # compute pairwise distances
        pdists = distance.squareform(distance.pdist(data[:, :3]) <= 1.5)
        pdists = torch.from_numpy(pdists)
        edge_index, edge_attr = dense_to_sparse(pdists)

        graph_data = Data(
            x=data,
            node_hvs=node_hvs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=data.shape[0],
            y=self.label_dict[pdbid],
        )

        # convert to pytorch geometric object, use the torch_geometric.utils.k_hop_subgraph function

        subgraph_1_hop_list = []
        subgraph_2_hop_list = []
        subgraph_3_hop_list = []
        for node_idx in range(graph_data.num_nodes):
            node_subset_1_hop, _, _, _ = k_hop_subgraph(
                [node_idx],
                num_hops=1,
                edge_index=edge_index,
                relabel_nodes=False,
                directed=False,
                num_nodes=graph_data.num_nodes,
            )
            node_subset_2_hop, _, _, _ = k_hop_subgraph(
                [node_idx],
                num_hops=2,
                edge_index=edge_index,
                relabel_nodes=False,
                directed=False,
                num_nodes=graph_data.num_nodes,
            )
            node_subset_3_hop, _, _, _ = k_hop_subgraph(
                [node_idx],
                num_hops=3,
                edge_index=edge_index,
                relabel_nodes=False,
                directed=False,
                num_nodes=graph_data.num_nodes,
            )
            # the hypervectors should be polarized after the summation
            hv_1_hop = (
                graph_data.node_hvs[node_subset_1_hop].sum(dim=0).unsqueeze(dim=0)
            )
            hv_2_hop = (
                graph_data.node_hvs[node_subset_2_hop].sum(dim=0).unsqueeze(dim=0)
            )
            hv_3_hop = (
                graph_data.node_hvs[node_subset_3_hop].sum(dim=0).unsqueeze(dim=0)
            )
            # why just two hops? we can specify this as a parameter
            subgraph_1_hop_list.append(hv_1_hop)
            subgraph_2_hop_list.append(hv_2_hop)
            subgraph_3_hop_list.append(hv_3_hop)

        # graph_data.subgraph_1_hop_hvs = torch.cat(subgraph_1_hop_list)
        # graph_data.subgraph_2_hop_hvs = torch.cat(subgraph_2_hop_list)

        subgraph_1_hop_hvs = torch.cat(subgraph_1_hop_list)
        subgraph_2_hop_hvs = torch.cat(subgraph_2_hop_list)
        subgraph_3_hop_hvs = torch.cat(subgraph_3_hop_list)

        # need to make the phi vectors orthogonal

        phi = (2 * torch.eye(n=4, m=self.D)) - 1

        phi_node_feat = phi[0, :]
        phi_1_hop = phi[1, :]
        phi_2_hop = phi[2, :]
        phi_3_hop = phi[3, :]

        #
        graph_data.graph_hvs = (
            (graph_data.node_hvs * phi_node_feat)
            + (phi_1_hop * subgraph_1_hop_hvs)
            + (phi_2_hop * subgraph_2_hop_hvs)
            + (phi_3_hop * subgraph_3_hop_hvs)
        ).sum(dim=0)

        torch.save(graph_data, cache_file)

        return graph_data


def job(pdbid_tup: tuple, featurizer: HDModel):
    pdbid, pdbid_df = pdbid_tup
    pocket_f = Path(
        f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_pocket.pdb"
    )
    ligand_f = Path(
        f"/p/lustre2/jones289/data/raw_data/v2016/{pdbid}/{pdbid}_ligand.mol2"
    )

    try:
        data = featurizer.featurize(protein_file=pocket_f, ligand_file=ligand_f)

        affinity = pdbid_df["-logKd/Ki"].values[:]
        label = affinity > 6 or affinity < 4
        return (data, label)

    except Exception as e:
        print(e)
        return


# def train(model, hv_train, y_train, epochs=10):
def train(model, dataloader, model_name, epochs=10):
    # import ipdb
    # ipdb.set_trace()

    if model_name == "complex-graph":
        single_pass_train_start = time.perf_counter()
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="building associative memory"
        ):
            model.update_am(
                dataset_hvs=batch.graph_hvs.reshape(-1, model.D), labels=batch.y
            )
            # pass
        single_pass_train_time = time.perf_counter() - single_pass_train_start

        print(f"retraining took {single_pass_train_time} seconds")

        # model.build_am(hv_train, y_train)

        learning_curve_list = []

        retrain_start = time.perf_counter()
        for _ in range(epochs):
            for batch in tqdm(
                dataloader, total=len(dataloader), desc="perceptron training"
            ):
                mistake_ct = model.retrain(
                    dataset_hvs=batch.graph_hvs.reshape(-1, model.D), labels=batch.y
                )
                # mistake_ct = model.retrain(hv_train, y_train, return_mistake_count=True)
                learning_curve_list.append(mistake_ct)

        retrain_time = time.perf_counter() - retrain_start

        print(
            f"training took {retrain_time} seconds (avg. {retrain_time/epochs} sec. per epoch)"
        )
        return learning_curve_list, single_pass_train_time, retrain_time

    else:
        single_pass_train_start = time.perf_counter()
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="building associative memory"
        ):
            data, y = batch
            model.update_am(dataset_hvs=data, labels=y)
            # pass
        single_pass_train_time = time.perf_counter() - single_pass_train_start

        print(f"retraining took {single_pass_train_time} seconds")

        # model.build_am(hv_train, y_train)

        learning_curve_list = []

        retrain_start = time.perf_counter()
        for _ in range(epochs):
            for batch in tqdm(
                dataloader, total=len(dataloader), desc="perceptron training"
            ):
                data, y = batch
                mistake_ct = model.retrain(dataset_hvs=data, labels=y)
                # mistake_ct = model.retrain(hv_train, y_train, return_mistake_count=True)
                learning_curve_list.append(mistake_ct)

        retrain_time = time.perf_counter() - retrain_start

        print(
            f"training took {retrain_time} seconds (avg. {retrain_time/epochs} sec. per epoch)"
        )
        return learning_curve_list, single_pass_train_time, retrain_time


def test(model, dataloader, model_name):
    pred_time_list = []
    conf_time_list = []
    pred_list = []
    conf_list = []
    true_list = []

    if model_name == "complex-graph":
        for batch in tqdm(dataloader, total=len(dataloader), desc="testing"):
            true_list.append(batch.y)

            pred_start = time.perf_counter()
            pred = model.predict(batch.graph_hvs.reshape(-1, model.D))
            pred_time = time.perf_counter() - pred_start
            pred_list.append(pred)
            pred_time_list.append(pred_time)

            conf_start = time.perf_counter()
            conf = model.compute_confidence(batch.graph_hvs.reshape(-1, model.D))
            conf_time = time.perf_counter() - conf_start
            conf_list.append(conf)
            conf_time_list.append(conf_time)
    else:
        # import ipdb
        # ipdb.set_trace()
        for batch in tqdm(dataloader, total=len(dataloader), desc="testing"):
            data, y = batch
            true_list.append(y)

            pred_start = time.perf_counter()
            pred = model.predict(data.squeeze())
            pred_time = time.perf_counter() - pred_start
            pred_list.append(pred)
            pred_time_list.append(pred_time)

            conf_start = time.perf_counter()
            conf = model.compute_confidence(data.squeeze())
            conf_time = time.perf_counter() - conf_start
            conf_list.append(conf)
            conf_time_list.append(conf_time)

    print(
        f"testing took sum of conf:{np.sum(conf_time_list)} + pred: {np.sum(pred_time_list)} seconds, mean (per-batch) time conf:{np.mean(conf_time_list)}, pred:{(pred_time_list)}"
    )

    return {
        "y_pred": torch.cat(pred_list),
        "y_true": torch.cat(true_list),
        "eta": torch.cat(conf_list),
        "test_time": pred_time_list,
        "conf_test_time": conf_time_list,
    }


def complex_graph_main():
    # todo: switch v2016 to updated pdbbind version

    train_list = []
    for split in ["general_train", "refined_train"]:
        train_dataset = ComplexGraphHDDataset(
            data_dir=Path(args.data_dir),
            meta_path=Path(args.meta_path),
            node_feat_size=22,
            p=args.p,
            split=split,
            D=args.D,
        )
        train_list.append(train_dataset)

    from torch.utils.data import ConcatDataset

    train_dataset = ConcatDataset(train_list)

    test_list = []
    for split in ["core_test"]:
        test_dataset = ComplexGraphHDDataset(
            data_dir=Path(args.data_dir),
            meta_path=Path(args.meta_path),
            D=args.D,
            node_feat_size=22,
            p=args.p,
            split=split,
        )
        test_list.append(test_dataset)

    test_dataset = ConcatDataset(test_list)

    train_dataloader = DataLoader(
        train_dataset, num_workers=32, batch_size=128, persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset, num_workers=32, batch_size=128, persistent_workers=True
    )

    model = HDModel(D=10000)

    train(
        model=model,
        dataloader=train_dataloader,
        model_name=args.model,
        epochs=args.epochs,
    )

    result_dict = test(model=model, dataloader=test_dataloader, model_name=args.model)

    print(
        f"roc_auc: {roc_auc_score(y_true=result_dict['y_true'], y_score=result_dict['eta'])}"
    )
    print(
        classification_report(
            y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]
        )
    )

    import time

    ts = time.perf_counter()
    with open(f"{args.model}-{args.seed}-result_dict_{ts}.pkl", "wb") as handle:
        pickle.dump(result_dict, handle)


def aa_seq_ecfp_main():
    print("working on dataset")
    # DisjointComplexDataset(data_dir=args.data_dir, meta_path=args.meta_path, D=args.D, p=args.p, split="train")

    train_list = []
    for split in ["general_train", "refined_train"]:
        # train_dataset = ComplexGraphHDDataset(data_dir=Path(args.data_dir),
        #  meta_path=Path(args.meta_path), node_feat_size=22, p=args.p, split=split, D=args.D)
        train_dataset = DisjointComplexDataset(
            data_dir=args.data_dir,
            meta_path=args.meta_path,
            D=args.D,
            p=args.p,
            split=split,
        )
        train_list.append(train_dataset)

    train_dataset = ConcatDataset(train_list)

    test_list = []
    for split in ["core_test"]:
        # test_dataset = ComplexGraphHDDataset(data_dir=Path(args.data_dir),
        #  meta_path=Path(args.meta_path),
        #  D=args.D, node_feat_size=22, p=args.p, split=split)

        test_dataset = DisjointComplexDataset(
            data_dir=args.data_dir,
            meta_path=args.meta_path,
            D=args.D,
            p=args.p,
            split=split,
        )
        test_list.append(test_dataset)

    test_dataset = ConcatDataset(test_list)

    train_dataloader = DataLoader(
        train_dataset, num_workers=32, batch_size=128, persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset, num_workers=32, batch_size=128, persistent_workers=True
    )

    model = HDModel(D=10000)

    train(
        model=model,
        dataloader=train_dataloader,
        model_name=args.model,
        epochs=args.epochs,
    )

    result_dict = test(model=model, dataloader=test_dataloader, model_name=args.model)

    print(
        f"roc_auc: {roc_auc_score(y_true=result_dict['y_true'], y_score=result_dict['eta'])}"
    )
    print(
        classification_report(
            y_pred=result_dict["y_pred"], y_true=result_dict["y_true"]
        )
    )

    import time

    ts = time.perf_counter()
    with open(f"{args.model}-{args.seed}-result_dict_{ts}.pkl", "wb") as handle:
        pickle.dump(result_dict, handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, help="dimension of hypervector")
    parser.add_argument("--p", type=float, help="proportion of data to use")
    parser.add_argument("--seed", type=int, help="seed for rng", default=0)
    parser.add_argument(
        "--epochs", type=int, help="number of training epochs", default=10
    )
    parser.add_argument("--model", choices=["aa_seq_ecfp", "complex-graph"])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/p/lustre2/jones289/data/raw_data/v2016")
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path(
            "/g/g13/jones289/workspace/fast_md/data/metadata/pdbbind_2016_train_val_test.csv"
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # and a bunch of other stuff

    if args.model == "aa_seq_ecfp":
        aa_seq_ecfp_main()
    elif args.model == "complex-graph":
        complex_graph_main()

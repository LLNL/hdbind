import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import csv
import math

# Note: this example requires the torch_geometric library: https://pytorch-geometric.readthedocs.io
import torch_geometric

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics

from torchhd import functional
from torchhd import embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # hypervectors dimension

# for other available datasets see: https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html?highlight=tudatasets

from collections import defaultdict
from rdkit import Chem

import logging
logger = logging.getLogger(__name__)

import warnings

import pandas as pd

class SMILESGraphDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, smiles_field:str, target_fields:list, num_classes=2) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.smiles_field = smiles_field
        self.target_fields = target_fields


        self.smiles_list = []
        self.targets = []
        self.data = []
        self.targets = []



        self.load_csv()


        self.num_classes = num_classes

    def load_smiles(self, smiles_list, targets):
        """
        Load the dataset from SMILES and targets.

        Parameters:
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            transform (Callable, optional): data transformation function
            lazy (bool, optional): if lazy mode is used, the molecules are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """


        num_sample = len(smiles_list)
        if num_sample > 1000000:
            warnings.warn("Preprocessing molecules of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_smiles(lazy=True) to construct molecules in the dataloader instead.")
        # for field, target_list in targets.items():
            # if len(target_list) != num_sample:
                # raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                #  "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.smiles_list = []
        self.data = []
        self.targets = defaultdict(list)

        # import ipdb 
        # ipdb.set_trace()

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % smiles)
                continue
            # mol = data.Molecule.from_molecule(mol)
            mol = torch_geometric.utils.smiles.from_smiles(smiles)
            
            self.data.append(mol)
            self.smiles_list.append(smiles)
            for field in targets:
                self.targets[field].append(targets[field][i])



    def load_csv(self):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        print(f"target_fields: {self.target_fields}")
        if self.target_fields is not None:
            self.target_fields = set(self.target_fields)


        df = pd.read_csv(self.csv_path)

        smiles = df[self.smiles_field].values.tolist()
        targets = {target: df[target].values.tolist() for target in self.target_fields}


        self.load_smiles(smiles, targets)



    def __len__(self):

        return len(self.smiles_list)
        

    def __getitem__(self, idx):

        self.data[idx].y = torch.cat([torch.tensor(self.targets[field][idx]).reshape(-1,1) for field in self.target_fields], dim=1)
        return self.data[idx]


# import ipdb
# ipdb.set_trace()
graphs = SMILESGraphDataset("/usr/WS2/bcwc/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv", smiles_field="rdkitSmiles", target_fields=["p_np"])

train_size = int(0.7 * len(graphs))
test_size = len(graphs) - train_size
train_ld, test_ld = torch.utils.data.random_split(graphs, [train_size, test_size])


def sparse_stochastic_graph(G):
    """
    Returns a sparse adjacency matrix of the graph G.
    The values indicate the probability of leaving a vertex.
    This means that each column sums up to one.
    """
    rows, columns = G.edge_index
    # Calculate the probability for each column
    values_per_column = 1.0 / torch.bincount(columns, minlength=G.num_nodes)
    values_per_node = values_per_column[columns]
    size = (G.num_nodes, G.num_nodes)
    return torch.sparse_coo_tensor(G.edge_index, values_per_node, size)


def pagerank(G, alpha=0.85, max_iter=100, tol=1e-06):
    N = G.num_nodes
    M = sparse_stochastic_graph(G) * alpha
    v = torch.zeros(N, device=G.edge_index.device) + 1 / N
    p = torch.zeros(N, device=G.edge_index.device) + 1 / N
    for _ in range(max_iter):
        v_prev = v
        v = M @ v + p * (1 - alpha)

        err = (v - v_prev).abs().sum()
        if tol != None and err < N * tol:
            return v
    return v


def to_undirected(edge_index):
    """
    Returns the undirected edge_index
    [[0, 1], [1, 0]] will result in [[0], [1]]
    """
    edge_index = edge_index.sort(dim=0)[0]
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def min_max_graph_size(graph_dataset):
    if len(graph_dataset) == 0:
        return None, None

    max_num_nodes = float("-inf")
    min_num_nodes = float("inf")

    for G in graph_dataset:
        num_nodes = G.num_nodes
        max_num_nodes = max(max_num_nodes, num_nodes)
        min_num_nodes = min(min_num_nodes, num_nodes)

    return min_num_nodes, max_num_nodes


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.node_ids = embeddings.Random(size, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):

        import pdb 
        pdb.set_trace()

        pr = pagerank(x)
        pr_sort, pr_argsort = pr.sort()

        node_id_hvs = torch.zeros((x.num_nodes, DIMENSIONS), device=device)
        node_id_hvs[pr_argsort] = self.node_ids.weight[: x.num_nodes]

        row, col = to_undirected(x.edge_index)

        hvs = functional.bind(node_id_hvs[row], node_id_hvs[col])
        return functional.multiset(hvs)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


min_graph_size, max_graph_size = min_max_graph_size(graphs)
model = Model(graphs.num_classes, max_graph_size)
model = model.to(device)

with torch.no_grad():
    for samples in tqdm(train_ld, desc="Training"):
        samples.edge_index = samples.edge_index.to(device)
        samples.y = samples.y.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[samples.y] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

accuracy = torchmetrics.Accuracy()

with torch.no_grad():
    for samples in tqdm(test_ld, desc="Testing"):
        samples.edge_index = samples.edge_index.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1).unsqueeze(0)

        # import ipdb
        # ipdb.set_trace()

        accuracy.update(predictions.cpu().reshape(-1,1), samples.y.reshape(-1,1))

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

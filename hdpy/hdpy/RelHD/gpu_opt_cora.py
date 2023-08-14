import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
import numpy as np
from itertools import permutations
import datetime
from torch_geometric.transforms import NormalizeFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from timeit import default_timer as timer
from scipy.sparse import csr_matrix
from collections import defaultdict, OrderedDict
# import cuda.hd_enc

dataset = Planetoid(root='./data/Cora', name='Cora')
#dataset = Planetoid(root='./data/CiteSeer', name='CiteSeer')
#dataset = Planetoid(root='./data/Pubmed', name='Pubmed')

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)


print(dataset, "length of dataset: ", len(dataset))
data = dataset[0]
print("Data: ", data)
print("# of node features: ", data.num_node_features)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of val nodes: {data.val_mask.sum()}')
print(f'Number of test nodes: {data.test_mask.sum()}')
print("Data Undirected?: ", data.is_undirected())
print(data.x[0].to_sparse().indices())
print(data.x[0].to_sparse().values())

arr = torch.flatten(data.x)
max_feature_val = arr.max()
min_feature_val = arr.min()
print('[Feature Val] Max: {0} and Min: {1}'.format(max_feature_val, min_feature_val))


def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))

def gathering(input, node_num):
    result = defaultdict(list)
    for i in input:
        result[i[1]].append(i[0])
    flattened_gathering = []
    flattened_gathering_indptr = [0]
    curr_size = 0
    for node_idx in range(node_num):
        v = result[node_idx]
        if len(v) == 0:
            flattened_gathering_indptr.append(curr_size)
        else:
            flattened_gathering.extend(v)
            curr_size += len(v)
            flattened_gathering_indptr.append(curr_size)
    return flattened_gathering, flattened_gathering_indptr

def gen_hv(D, F, non_linear=False):
    mu = 0
    sigma = 1
    hv_mat = torch.empty(F, D, dtype=torch.float32, device=device)
    hv_mat.normal_(mu, sigma)
    if not non_linear:
        hv_mat = polarize(hv_mat, True)
    return hv_mat

def gen_lvs(D: int, Q: int):
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list()
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(list(li))
    return levels


def flip_based_lvs(D: int, totalFeatures: int, flip_factor: int, non_linear=False):

    nFlip = int(D//flip_factor)

    mu = 0
    sigma = 1
    bases = np.random.normal(mu, sigma, D)
    if non_linear == False:
        bases = 2*(bases >= 0) - 1  # polarize

    import copy
    generated_hvs = [copy.copy(bases)]

    for ii in range(totalFeatures-1):
        # id_hvs.append(np.random.normal(mu, sigma, D))
        idx_to_flip = np.random.randint(0, D, size=nFlip)
        bases[idx_to_flip] *= (-1)
        generated_hvs.append(copy.copy(bases))

    return generated_hvs

def polarize(arr, enable):
    if enable == True:
        arr = torch.sign(arr)
        arr[arr==0] = 1
    return arr

def rho(hv, num):
    if hv.dim() == 1:
        return torch.roll(hv, num)
    else:
        return torch.roll(hv, num, dims=1)

def normalize(x, x_test=None, normalizer='l2'):
    if normalizer == 'l2':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(norm='l2').fit(x)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)

    elif normalizer == 'minmax': # Here, we assume knowing the global max & min
        from sklearn.preprocessing import MinMaxScaler
        if x_test is None:
            x_data = x
        else:
            x_data = np.concatenate((x, x_test), axis=0)

        scaler = MinMaxScaler().fit(x_data)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)

    raise NotImplemented


dim=4096
print("DIM: ", dim)
binarize=False
enable_non_linear=False
refine_threshold = 0.5   # higher, more memorization
train_iter_num = 20
memoryhv_iter_num = 20
lr = 1
q = 100  # q = 1 this generates two levels, [0, 1]

if max_feature_val != 1:
    trace('Normalizing node features...')
    node_features, _ = normalize(data.x, normalizer='l2')
    node_features = torch.Tensor(node_features)

    arr = torch.flatten(node_features)
    max_feature_val = arr.max()
    min_feature_val = arr.min()
    print('[After Normalization: Feature Val] Max: {0} and Min: {1}'.format(max_feature_val, min_feature_val))
else:
    node_features = data.x
# node_features = data.x

assert(node_features.shape == data.x.shape)

###################
trace('Feature HV Generation')
feature_hvs = gen_hv(D=dim, F=data.num_node_features, non_linear=enable_non_linear)

###################
trace('Level HV Generation')
level_hvs = gen_lvs(D=dim, Q=q)
level_hvs = torch.Tensor(level_hvs).to(device)  # float32
all_feature_arange = torch.arange(data.num_node_features)

###################
trace('Node Property HV Generation')
boo0 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)
boo1 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)
boo2 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)

###################
trace('Node HV Generation')
csr_node_features = csr_matrix(node_features)
feature_indices = torch.Tensor(csr_node_features.indices).to(device).to(torch.int32)
feature_values = torch.Tensor(csr_node_features.data).to(device)
csr_info = torch.Tensor(csr_node_features.indptr).to(device).to(torch.int32)

if device.type == 'cuda':
    cuda_start.record()
else:
    start=timer()

## IMPLEMENTATION 1
node_hvs = torch.zeros((data.num_nodes, dim), dtype=torch.float32, device=device)
for node_idx, node in enumerate(node_features):
    feature_exist = node.to_sparse().coalesce().indices().squeeze()
    if feature_exist.ndim == 0:
        node_hvs[node_idx] = feature_hvs[feature_exist]
    else:
        node_hvs[node_idx] = torch.sum(feature_hvs[feature_exist], dim=0).view(-1)
    
    # lv_indices = (node.to_sparse().coalesce().values() * q).squeeze().to(torch.int64)
    # node_hvs[node_idx] = torch.sum(torch.mul(feature_hvs[feature_exist], level_hvs[lv_indices]), dim=0).view(-1)

## IMPLEMENTATION 2 (CUDA)
# node_hvs = cuda.hd_enc.id_wrapper(feature_hvs, feature_indices, csr_info, data.num_nodes, dim)  # actual encoding

## IMPLEMENTATION 2 (PACKING)
# np_id_hvs = feature_hvs.cpu().numpy()
# np_id_hvs[np_id_hvs < 0] = 0
# np_id_hvs = np_id_hvs.astype(np.int32)
# id_hvs_packed = torch.tensor(np.packbits(np_id_hvs, axis=-1), dtype=torch.uint8, device=device)
# node_hvs = cuda.hd_enc.id_packed_wrapper(id_hvs_packed, feature_indices, csr_info, data.num_nodes, dim)

if device.type == 'cuda':
    torch.cuda.synchronize()
    cuda_end.record()
    print("[STATS] Node HVs time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
else:
    end=timer()
    print("[STATS] Node HVs time: {}".format(end - start))
node_hvs = polarize(node_hvs, binarize)


edge_coordinates = torch.transpose(data.edge_index, 0, 1).cpu().detach().tolist()
gather_idx, gather_indptr = gathering(edge_coordinates, data.num_nodes)
gather_idx = torch.Tensor(gather_idx).to(torch.int32).to(device)
gather_indptr = torch.Tensor(gather_indptr).to(torch.int32).to(device)


###################
trace('Memory HV Generation')
if device.type == 'cuda':
    cuda_start.record()
else:
    start=timer()

# IMPLEMENTATION 1: CUDA
# memory_hvs = cuda.hd_enc.memory_hv_build(node_hvs, gather_idx, gather_indptr, data.num_nodes, dim)

# IMPLEMENTATION 2 (CPU) Works better than sort / gather coordinate-sum
memory_hvs = torch.zeros((data.num_nodes, dim), dtype=torch.float32, device=device)
for (h, t) in edge_coordinates:
    memory_hvs[t] += node_hvs[h]

if device.type == 'cuda':
    cuda_end.record()
    torch.cuda.synchronize()
    print("[STATS] Memory HVs time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
else:
    end=timer()
    print("[STATS] Memory HVs time: {}".format(end-start))

memory_hvs = polarize(memory_hvs, binarize)

###################
trace('Memory LV2 Generation')
start = timer()

if device.type == 'cuda':
    cuda_start.record()
else:
    start=timer()

# IMPLEMENTATION 1: CUDA
# memory_lv2_hvs = cuda.hd_enc.memory_hv_build(memory_hvs, gather_idx, gather_indptr, data.num_nodes, dim)


# IMPLEMENTATION 2 (CPU) Works better than sort / gather coordinate-sum
memory_lv2_hvs = torch.zeros((data.num_nodes, dim), dtype=torch.float32, device=device)
for (h, t) in edge_coordinates:
    memory_lv2_hvs[t] += memory_hvs[h]


if device.type == 'cuda':
    cuda_end.record()
    torch.cuda.synchronize()
    print("[STATS] Memory HVs LV2 time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
else:
    end=timer()
    print("[STATS] Memory HVs LV2 time: {}".format(end-start))

memory_lv2_hvs = rho(memory_lv2_hvs, -1)
memory_lv2_hvs = polarize(memory_lv2_hvs, binarize)


num_classes = len(torch.unique(data.y))
class_hvs = torch.zeros((num_classes, dim), dtype=torch.float32, device=device)
y_true = data.y[data.test_mask]
total = int(data.test_mask.sum())

###################
trace('Build Relation HVs')
rel_hvs = node_hvs*boo0 + memory_hvs*boo1 + memory_lv2_hvs*boo2

###################
trace('Single Pass Training')
# single pass training
train_node_idx = torch.where(data.train_mask==True)[0]
train_data = rel_hvs[train_node_idx]
train_labels = (data.y[train_node_idx])

start=timer()
for c in range(num_classes):
    t = torch.sum(train_data[train_labels==c], dim=0)
    class_hvs[c] = t

if device.type == 'cuda':    
    torch.cuda.synchronize()
end=timer()

# class_hvs = polarize(class_hvs, True)
print("[STATS] Train time: {}".format(end - start))

###################
trace('Testing => num_classes:', num_classes, "total test size:", total)
test_node_idx = torch.where(data.test_mask==True)[0]

start = timer()
if not binarize:
    sim_mat = (torch.mm(class_hvs, rel_hvs[test_node_idx].T).T/ torch.linalg.norm(class_hvs, dim=1, ord=2)).T
else:
    sim_mat = torch.mm(class_hvs, rel_hvs[test_node_idx].T)
pred_list = torch.argmax(sim_mat, axis=0)
end = timer()
print("[STATS] Test time: {}".format(end - start))
print(f1_score(y_true, pred_list.cpu().detach().numpy(), average='micro'))  # micro is same as acc


###################
trace('Iterative Training')
start = timer()
for _ in range(train_iter_num):
    # refine=0
    for ii, node_idx in enumerate(train_node_idx):
        sim_vec = torch.matmul(class_hvs, rel_hvs[node_idx])
        if not binarize:
            sim_vec = sim_vec/torch.linalg.norm(class_hvs, dim=1, ord=2)
        pred_y = torch.argmax(sim_vec)
        node_y = train_labels[ii]
        if pred_y != node_y:
            # refine+=1
            class_hvs[node_y] += lr * rel_hvs[node_idx]
            class_hvs[pred_y] -= lr * rel_hvs[node_idx]
    start1 = timer() 
    if not binarize:
        sim_mat = (torch.mm(class_hvs, rel_hvs[test_node_idx].T).T/ torch.linalg.norm(class_hvs, dim=1, ord=2)).T
    else:
        sim_mat = torch.mm(class_hvs, rel_hvs[test_node_idx].T)
    pred_list = torch.argmax(sim_mat, axis=0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end1 = timer()
    print("[STATS] Test time: {}".format(end1 - start1))
    print(f1_score(y_true, pred_list.cpu().detach().numpy(), average='micro'))  # micro is same as acc

class_hvs = polarize(class_hvs, True)
end = timer()
print("[STATS] Retrain per epoch avg time: {}".format((end - start)/train_iter_num))

# print(node_hvs.max())
# print(memory_hvs.max())
# print(memory_lv2_hvs.max())
# print(class_hvs.max())

trace('Testing => num_classes:', num_classes, "total test size:", total)

start = timer()
if not binarize:
    sim_mat = (torch.mm(class_hvs, rel_hvs[test_node_idx].T).T/ torch.linalg.norm(class_hvs, dim=1, ord=2)).T
else:
    sim_mat = torch.mm(class_hvs, rel_hvs[test_node_idx].T)
pred_list = torch.argmax(sim_mat, axis=0)
if device.type == 'cuda':
    torch.cuda.synchronize()
end = timer()
print("[STATS] Test time: {}".format(end - start))
print(f1_score(y_true, pred_list.cpu().detach().numpy(), average='micro'))  # micro is same as acc

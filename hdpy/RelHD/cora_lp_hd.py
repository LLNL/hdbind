import torch
from sklearn.metrics import roc_auc_score

import numpy as np
import datetime

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)


transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

dataset = Planetoid(root='./data/Cora', name='Cora', transform=transform)
# dataset = Planetoid(root='./data/CiteSeer', name='CiteSeer', transform=transform)
# dataset = Planetoid(root='./data/Pubmed', name='Pubmed', transform=transform)


# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.
train_data, val_data, test_data = dataset[0]

print("Data: ", train_data)
print("# of node features: ", train_data.num_node_features)
print("Data Undirected?: ", train_data.is_undirected())
print(train_data.x[0].to_sparse().indices())
print(train_data.x[0].to_sparse().values())

arr = torch.flatten(train_data.x)
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


def concatenate_hv(hv1, hv2):
    if hv1.dim() == 1:
         return torch.cat((hv1[:hv1.shape[0]//2], hv2[hv2.shape[0]//2:]))
    else:
        hv1 = hv1[:, :hv1.shape[1]//2]
        hv2 = hv2[:, hv2.shape[1]//2:]
        return torch.cat((hv1, hv2), dim=1)


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


dim=8192
print("DIM: ", dim)
binarize=False
enable_non_linear=False
q = 100  # q = 1 this generates two levels, [0, 1]

if max_feature_val != 1:
    trace('Normalizing node features...')
    node_features, _ = normalize(train_data.x.cpu(), normalizer='l2')
    node_features = torch.Tensor(node_features)

    arr = torch.flatten(node_features)
    max_feature_val = arr.max()
    min_feature_val = arr.min()
    print('[After Normalization: Feature Val] Max: {0} and Min: {1}'.format(max_feature_val, min_feature_val))
else:
    node_features = train_data.x
# node_features = data.x

assert(node_features.shape == train_data.x.shape)

###################
trace('Feature HV Generation')
feature_hvs = gen_hv(D=dim, F=train_data.num_node_features, non_linear=enable_non_linear)

###################
trace('Level HV Generation')
level_hvs = gen_lvs(D=dim, Q=q)
level_hvs = torch.Tensor(level_hvs).to(device)  # float32
all_feature_arange = torch.arange(train_data.num_node_features)

###################
trace('Node Property HV Generation')
boo0 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)
boo1 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)
boo2 = torch.Tensor(np.random.choice([1., -1.], size=dim, p=[0.5, 0.5]).astype(np.float32)).to(device)

###################
trace('Node HV Generation')

# if device.type == 'cuda':
#     cuda_start.record()
# else:
#     start=timer()

## IMPLEMENTATION 1
node_hvs = torch.zeros((train_data.num_nodes, dim), dtype=torch.float32, device=device)
for node_idx, node in enumerate(node_features):
    feature_exist = node.to_sparse().coalesce().indices().squeeze()
    if feature_exist.ndim == 0:
        node_hvs[node_idx] = feature_hvs[feature_exist]
    else:
        node_hvs[node_idx] = torch.sum(feature_hvs[feature_exist], dim=0).view(-1)
if enable_non_linear == True:
    node_hvs = torch.tanh(node_hvs)


# trace("Bloom Filter Encoding")
# hash_num = 64
# def bloom_encode(data, hv_dim):
#     hash_table = np.random.randint(0, hv_dim, (train_data.num_node_features, hash_num))
#     hash_table = torch.Tensor(hash_table).to(device).long()
#     bloom = torch.zeros((data.shape[0], hv_dim))
#     for data_idx, features in enumerate(data):
#         feature_exist = features.to_sparse().coalesce().indices().squeeze()
#         xx = hash_table[feature_exist].flatten().unique()
#         bloom[data_idx, xx] = 1
#     return bloom
# node_hvs = torch.Tensor(bloom_encode(node_features, dim)).to(device)
# node_hvs[node_hvs == 0] = -1

# if device.type == 'cuda':
#     torch.cuda.synchronize()
#     cuda_end.record()
#     print("[STATS] Node HVs time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
# else:
#     end=timer()
#     print("[STATS] Node HVs time: {}".format(end - start))
node_hvs = polarize(node_hvs, binarize)


edge_coordinates = torch.transpose(train_data.edge_index, 0, 1).cpu().detach().tolist()
gather_idx, gather_indptr = gathering(edge_coordinates, train_data.num_nodes)
gather_idx = torch.Tensor(gather_idx).to(torch.int32).to(device)
gather_indptr = torch.Tensor(gather_indptr).to(torch.int32).to(device)


###################
trace('Memory HV Generation')
# if device.type == 'cuda':
#     cuda_start.record()
# else:
#     start=timer()

# IMPLEMENTATION 1: CUDA
# memory_hvs = cuda.hd_enc.memory_hv_build(node_hvs, gather_idx, gather_indptr, data.num_nodes, dim)

# IMPLEMENTATION 2 (CPU) Works better than sort / gather coordinate-sum
memory_hvs = torch.zeros((train_data.num_nodes, dim), dtype=torch.float32, device=device)
for (h, t) in edge_coordinates:
    memory_hvs[t] += node_hvs[h]

# if device.type == 'cuda':
#     cuda_end.record()
#     torch.cuda.synchronize()
#     print("[STATS] Memory HVs time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
# else:
#     end=timer()
#     print("[STATS] Memory HVs time: {}".format(end-start))

memory_hvs = polarize(memory_hvs, binarize)


###################
trace('Memory LV2 Generation')
start = timer()

# if device.type == 'cuda':
#     cuda_start.record()
# else:
#     start=timer()

# IMPLEMENTATION 1: CUDA
# memory_lv2_hvs = cuda.hd_enc.memory_hv_build(memory_hvs, gather_idx, gather_indptr, data.num_nodes, dim)


# IMPLEMENTATION 2 (CPU) Works better than sort / gather coordinate-sum
memory_lv2_hvs = torch.zeros((train_data.num_nodes, dim), dtype=torch.float32, device=device)
for (h, t) in edge_coordinates:
    memory_lv2_hvs[t] += memory_hvs[h]

# if device.type == 'cuda':
#     cuda_end.record()
#     torch.cuda.synchronize()
#     print("[STATS] Memory HVs LV2 time: {}".format(cuda_start.elapsed_time(cuda_end)*0.001))
# else:
#     end=timer()
#     print("[STATS] Memory HVs LV2 time: {}".format(end-start))

memory_lv2_hvs = polarize(memory_lv2_hvs, binarize)

trace('Build Relation HVs')
rel_hvs = node_hvs*boo0 + memory_hvs*boo1 #+ memory_lv2_hvs*boo2
rel_hvs = polarize(rel_hvs, True)


##################################

# ## Trial 1: Build two HVs for connected/disconnected 
# trace('Trial 1')
# nepochs = 0
# # Train
# link_hvs = torch.zeros((2, dim), dtype=torch.float32, device=device)
# edge_label_coordinates = torch.transpose(train_data.edge_label_index, 0, 1).cpu().detach().tolist()
# neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
#                                    num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
# neg_edge_label_coordinates = torch.transpose(neg_edge_index, 0, 1).cpu().detach().tolist()

# for (h, t) in edge_label_coordinates:
#     link_hvs[1] += rel_hvs[h] * rel_hvs[t]
# for (h, t) in neg_edge_label_coordinates:
#     link_hvs[0] += rel_hvs[h] * rel_hvs[t]
# link_hvs = polarize(link_hvs, binarize)

# # Retrain
# tot_train = len(edge_label_coordinates) + len(neg_edge_label_coordinates)
# for epoch in range(nepochs):
#     correct = 0
#     for (h, t) in edge_label_coordinates:
#         # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#         sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#         connected = torch.argmax(sim)
#         if connected != 1:
#             link_hvs[1] += rel_hvs[h] * rel_hvs[t]
#             link_hvs[0] -= rel_hvs[h] * rel_hvs[t]
#         else:
#             correct += 1
#     for (h, t) in neg_edge_label_coordinates:
#         # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#         sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#         connected = torch.argmax(sim)
#         if connected != 0:
#             link_hvs[1] -= rel_hvs[h] * rel_hvs[t]
#             link_hvs[0] += rel_hvs[h] * rel_hvs[t]
#         else:
#             correct += 1
#     link_hvs = polarize(link_hvs, binarize)

#     val_correct = 0
#     val_edge_coordinates = torch.transpose(val_data.edge_label_index, 0, 1).cpu().detach().tolist()
#     for idx, (h, t) in enumerate(val_edge_coordinates):
#         # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#         sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#         if torch.argmax(sim) == val_data.edge_label[idx]:
#             val_correct += 1
#     print("[STATS] Epoch {}: Train accuracy: {} ({}/{}), \
#                              Val accuracy: {} ({}/{})".format(epoch, correct / tot_train, correct, tot_train,
#                                                               val_correct / len(val_edge_coordinates), val_correct, len(val_edge_coordinates)))

# # Test
# correct = 0
# train_edge_coordinates = torch.transpose(train_data.edge_label_index, 0, 1).cpu().detach().tolist()
# for idx, (h, t) in enumerate(train_edge_coordinates):
#     # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#     sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#     if torch.argmax(sim) == train_data.edge_label[idx]:
#         correct += 1
# print("[STATS] Train (pos) accuracy: {} ({}/{})".format(correct / len(train_edge_coordinates), correct, len(train_edge_coordinates)))

# correct = 0
# val_edge_coordinates = torch.transpose(val_data.edge_label_index, 0, 1).cpu().detach().tolist()
# for idx, (h, t) in enumerate(val_edge_coordinates):
#     # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#     sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#     if torch.argmax(sim) == val_data.edge_label[idx]:
#         correct += 1
# print("[STATS] Val accuracy: {} ({}/{})".format(correct / len(val_edge_coordinates), correct, len(val_edge_coordinates)))

# pos_tot = 0
# pos_correct = 0
# correct = 0
# sims = []
# not_sims = []
# test_edge_coordinates = torch.transpose(test_data.edge_label_index, 0, 1).cpu().detach().tolist()
# for idx, (h, t) in enumerate(test_edge_coordinates):
#     # measure cosine similarity between link_hvs and rel_hvs[h] * rel_hvs[t]
#     sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=1)
#     if test_data.edge_label[idx] == 1:
#         sims.append(torch.max(sim).cpu().numpy())
#     else:
#         not_sims.append(torch.max(sim).cpu().numpy())

#     if torch.argmax(sim) == test_data.edge_label[idx]:
#         correct += 1
#     if test_data.edge_label[idx] == 1:
#         pos_tot += 1
#         if torch.argmax(sim) == test_data.edge_label[idx]:
#             pos_correct += 1
# neg_tot = len(test_edge_coordinates) - pos_tot
# print("[STATS] Test accuracy: {} ({}/{})".format(correct / len(test_edge_coordinates), correct, len(test_edge_coordinates)))
# print("[STATS] Test accuracy (Pos): {} ({}/{})".format(pos_correct / pos_tot, pos_correct, pos_tot))
# print("[STATS] Test accuracy (Neg): {} ({}/{})".format((correct - pos_correct) / neg_tot, correct - pos_correct, neg_tot))

# y_pred_prob = sims + not_sims
# y_true = [1] * len(sims) + [0] * len(not_sims)
# fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
# auc = roc_auc_score(y_true, y_pred_prob)
# print("AUC: {}".format(auc))
# plt.figure()
# plt.plot(fpr, tpr, label='AUC: %.2f' % auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.savefig('roc.png')




# ##################################

### Trial 2: Build one HV for connected/disconnected
trace('Trial 2')
neg_refine_threshold = -0.25
pos_refine_threshold = 0.25
threshold = 0
nepochs = 20

# Train
link_hvs = torch.zeros((dim), dtype=torch.float32, device=device)
edge_label_coordinates = torch.transpose(train_data.edge_label_index, 0, 1).cpu().detach().tolist()
neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                                   num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
neg_edge_label_coordinates = torch.transpose(neg_edge_index, 0, 1).cpu().detach().tolist()
len_pos = len(edge_label_coordinates)
len_neg = len(neg_edge_label_coordinates)
tot_train = len_neg + len_pos

pos_edge_hvs = torch.zeros((len_pos, dim), dtype=torch.float32, device=device)
neg_edge_hvs = torch.zeros((len(neg_edge_label_coordinates), dim), dtype=torch.float32, device=device)

trace('Build Edge HVs')
for ii, (h, t) in enumerate(edge_label_coordinates):
    pos_edge_hvs[ii] = rel_hvs[h] * rel_hvs[t]
for ii, (h, t) in enumerate(neg_edge_label_coordinates):
    neg_edge_hvs[ii] = rel_hvs[h] * rel_hvs[t]

trace('Single pass training')
for ii in range(len_pos):
    link_hvs += pos_edge_hvs[ii]
for ii in range(len_neg):
    link_hvs -= neg_edge_hvs[ii]
link_hvs = polarize(link_hvs, binarize)

print(link_hvs)
print(pos_edge_hvs)
print(neg_edge_hvs)

# Retrain
trace('Retrain')
for epoch in tqdm(range(nepochs)):
    for ii in range(len_pos):
        sim = torch.nn.functional.cosine_similarity(link_hvs, pos_edge_hvs[ii], dim=0)
        connected = 1 if sim > pos_refine_threshold else 0
        if connected != 1:
            link_hvs += pos_edge_hvs[ii]
    for ii in range(len_neg):
        sim = torch.nn.functional.cosine_similarity(link_hvs, neg_edge_hvs[ii], dim=0)
        connected = 1 if sim > neg_refine_threshold else 0
        if connected != 0:
            link_hvs -= neg_edge_hvs[ii]
    link_hvs = polarize(link_hvs, binarize)


test_edge_coordinates = torch.transpose(test_data.edge_label_index, 0, 1).cpu().detach().tolist()
sims = []
not_sims = []
for idx, (h, t) in enumerate(test_edge_coordinates):
    sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=0)
    if test_data.edge_label[idx] == 1:
        sims.append(sim.cpu().numpy())
    else:
        not_sims.append(sim.cpu().numpy())

# draw sims historgram
plt.figure()
plt.hist(sims, bins=100, color='b', alpha=0.5, label='sims')
plt.hist(not_sims, bins=100, color='r', alpha=0.5, label='not_sims')
plt.legend(loc='upper right')
plt.savefig('sims.png')

print("Find optimal threshold and play with it...")

y_pred_prob = sims + not_sims
y_true = [1] * len(sims) + [0] * len(not_sims)

def find_threshold(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr, tpr, auc, optimal_threshold

fpr, tpr, auc, new_thres = find_threshold(y_true, y_pred_prob)
pos_tot = 0
pos_correct = 0
correct = 0
test_edge_coordinates = torch.transpose(test_data.edge_label_index, 0, 1).cpu().detach().tolist()
for idx, (h, t) in enumerate(test_edge_coordinates):
    sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=0)
    connected = 1 if sim > new_thres else 0
    if connected == test_data.edge_label[idx]:
        correct += 1
    if test_data.edge_label[idx] == 1:
        pos_tot += 1
        if connected == test_data.edge_label[idx]:
            pos_correct += 1
neg_tot = len(test_edge_coordinates) - pos_tot
print("[STATS] Test accuracy: {} ({}/{})".format(correct / len(test_edge_coordinates), correct, len(test_edge_coordinates)))
print("[STATS] Test accuracy (Pos): {} ({}/{})".format(pos_correct / pos_tot, pos_correct, pos_tot))
print("[STATS] Test accuracy (Neg): {} ({}/{})".format((correct - pos_correct) / neg_tot, correct - pos_correct, neg_tot))

print("AUC: {}, Threshold: {}".format(auc, new_thres))
plt.figure()
plt.plot(fpr, tpr, label='AUC: %.2f' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc.png')




##################################

# ### Trial 3: Trial 2 + Batched processing on Retrain
# trace('Trial 3')
# neg_refine_threshold = -0.25
# pos_refine_threshold = 0.25
# threshold = 0
# nepochs = 20

# # Train
# link_hvs = torch.zeros((dim), dtype=torch.float32, device=device)
# edge_label_coordinates = torch.transpose(train_data.edge_label_index, 0, 1).cpu().detach().tolist()
# neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
#                                    num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
# neg_edge_label_coordinates = torch.transpose(neg_edge_index, 0, 1).cpu().detach().tolist()
# len_pos = len(edge_label_coordinates)
# len_neg = len(neg_edge_label_coordinates)
# tot_train = len_neg + len_pos

# pos_edge_hvs = torch.zeros((len_pos, dim), dtype=torch.float32, device=device)
# neg_edge_hvs = torch.zeros((len(neg_edge_label_coordinates), dim), dtype=torch.float32, device=device)

# trace('Build Edge HVs')
# for ii, (h, t) in enumerate(edge_label_coordinates):
#     pos_edge_hvs[ii] = rel_hvs[h] * rel_hvs[t]
# for ii, (h, t) in enumerate(neg_edge_label_coordinates):
#     neg_edge_hvs[ii] = rel_hvs[h] * rel_hvs[t]

# trace('Single pass training')
# for ii in range(len_pos):
#     link_hvs += pos_edge_hvs[ii]
# for ii in range(len_neg):
#     link_hvs -= neg_edge_hvs[ii]
# link_hvs = polarize(link_hvs, binarize)

# print(link_hvs)
# print(pos_edge_hvs)
# print(neg_edge_hvs)


# Retrain
# trace('Retrain')
# train_data_label_pos = torch.ones(len_pos, dtype=torch.float32, device=device)
# train_data_label_neg = torch.zeros(len_neg, dtype=torch.float32, device=device)
# for epoch in tqdm(range(nepochs)):
#     sim = torch.nn.functional.cosine_similarity(pos_edge_hvs, link_hvs, dim=1)
#     pos_connected = torch.where(sim > pos_refine_threshold, torch.ones(len_pos, dtype=torch.float32, device=device), torch.zeros(len_pos, dtype=torch.float32, device=device))
#     pos_check = (pos_connected != 1)
#     link_hvs += torch.sum(pos_edge_hvs[pos_check], dim=0)

#     sim = torch.nn.functional.cosine_similarity(neg_edge_hvs, link_hvs, dim=1)
#     neg_connected = torch.where(sim > neg_refine_threshold, torch.ones(len_neg, dtype=torch.float32, device=device), torch.zeros(len_neg, dtype=torch.float32, device=device))
#     neg_check = (neg_connected != 0)
#     link_hvs -= torch.sum(neg_edge_hvs[neg_check], dim=0)

#     # FIXME

#     print("Epoch: {}, Pos: {}, Neg: {}".format(epoch, torch.sum(pos_check)/len_pos, torch.sum(neg_check)/len_neg))

#     # for ii in range(len_pos):
#     #     sim = torch.nn.functional.cosine_similarity(link_hvs, pos_edge_hvs[ii], dim=0)
#     #     connected = 1 if sim > pos_refine_threshold else 0
#     #     if connected != 1:
#     #         link_hvs += pos_edge_hvs[ii]
#     # for ii in range(len_neg):
#     #     sim = torch.nn.functional.cosine_similarity(link_hvs, neg_edge_hvs[ii], dim=0)
#     #     connected = 1 if sim > neg_refine_threshold else 0
#     #     if connected != 0:
#     #         link_hvs -= neg_edge_hvs[ii]
#     link_hvs = polarize(link_hvs, binarize)



# test_edge_coordinates = torch.transpose(test_data.edge_label_index, 0, 1).cpu().detach().tolist()
# sims = []
# not_sims = []
# for idx, (h, t) in enumerate(test_edge_coordinates):
#     sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=0)
#     if test_data.edge_label[idx] == 1:
#         sims.append(sim.cpu().numpy())
#     else:
#         not_sims.append(sim.cpu().numpy())

# # draw sims historgram
# plt.figure()
# plt.hist(sims, bins=100, color='b', alpha=0.5, label='sims')
# plt.hist(not_sims, bins=100, color='r', alpha=0.5, label='not_sims')
# plt.legend(loc='upper right')
# plt.savefig('sims.png')

# print("Find optimal threshold and play with it...")

# y_pred_prob = sims + not_sims
# y_true = [1] * len(sims) + [0] * len(not_sims)

# def find_threshold(y_true, y_pred_prob):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
#     auc = roc_auc_score(y_true, y_pred_prob)

#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     return fpr, tpr, auc, optimal_threshold

# fpr, tpr, auc, new_thres = find_threshold(y_true, y_pred_prob)
# pos_tot = 0
# pos_correct = 0
# correct = 0
# test_edge_coordinates = torch.transpose(test_data.edge_label_index, 0, 1).cpu().detach().tolist()
# for idx, (h, t) in enumerate(test_edge_coordinates):
#     sim = torch.nn.functional.cosine_similarity(link_hvs, rel_hvs[h] * rel_hvs[t], dim=0)
#     connected = 1 if sim > new_thres else 0
#     if connected == test_data.edge_label[idx]:
#         correct += 1
#     if test_data.edge_label[idx] == 1:
#         pos_tot += 1
#         if connected == test_data.edge_label[idx]:
#             pos_correct += 1
# neg_tot = len(test_edge_coordinates) - pos_tot
# print("[STATS] Test accuracy: {} ({}/{})".format(correct / len(test_edge_coordinates), correct, len(test_edge_coordinates)))
# print("[STATS] Test accuracy (Pos): {} ({}/{})".format(pos_correct / pos_tot, pos_correct, pos_tot))
# print("[STATS] Test accuracy (Neg): {} ({}/{})".format((correct - pos_correct) / neg_tot, correct - pos_correct, neg_tot))

# print("AUC: {}, Threshold: {}".format(auc, new_thres))
# plt.figure()
# plt.plot(fpr, tpr, label='AUC: %.2f' % auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.savefig('roc.png')

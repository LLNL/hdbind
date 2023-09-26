import torch
import hd_enc_cuda

def packing(orig_vecs, N, D):
    packed_result = hd_enc_cuda.packing(orig_vecs, N, D)
    return packed_result

def lvid_wrapper(level_hvs, id_hvs, feature_indices, feature_values, csr_info, N, Q, D):
    hv_result = hd_enc_cuda.lvid(level_hvs, id_hvs,feature_indices, feature_values, csr_info, N, Q, D)
    return hv_result

def id_wrapper(id_hvs, feature_indices, csr_info, N, D):
    hv_result = hd_enc_cuda.id_enc(id_hvs, feature_indices, csr_info, N, D)
    return hv_result

def id_packed_wrapper(id_hvs_packed, feature_indices, csr_info, N, D):
    hv_result = hd_enc_cuda.id_packed_enc(id_hvs_packed, feature_indices, csr_info, N, D)
    return hv_result

def dense_idlv(level_hvs_packed, feature_hvs_packed, node_features, N, Q, F, D):
    hv_result = hd_enc_cuda.dense_idlv(level_hvs_packed, feature_hvs_packed, node_features, N, Q, F, D)
    return hv_result

def memory_hv_build(node_hvs, idx, idx_ptr, N, D):
    mhv_result = hd_enc_cuda.memory_hv_enc(node_hvs, idx, idx_ptr, N, D)
    return mhv_result

def memory_hv_lv2_build(node_hvs, idx, idx_ptr, N, D):
    mhv_result = hd_enc_cuda.memory_hv_lv2_enc(node_hvs, idx, idx_ptr, N, D)
    return mhv_result

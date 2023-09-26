#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor packing_cuda(
    torch::Tensor orig_vec,
    const int N, const int D);
torch::Tensor hd_enc_lvid_cuda(
    torch::Tensor level_hvs,
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor feature_values,
    torch::Tensor csr_info,
    const int N, const int Q, const int D);
torch::Tensor dense_idlv_cuda(
    torch::Tensor level_hvs_packed,
    torch::Tensor id_hvs_packed,
    torch::Tensor raw_data,
    const int N, const int Q, const int F, const int D);
torch::Tensor hd_enc_id_cuda(
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D);
torch::Tensor hd_enc_id_packed_cuda(
    torch::Tensor id_hvs_packed,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D);
torch::Tensor memory_hv_enc_cuda(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D);
torch::Tensor memory_hv_lv2_enc_cuda(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor packing(
    torch::Tensor orig_vec,
    const int N, const int D) {
    
    CHECK_INPUT(orig_vec);

    return packing_cuda(orig_vec, N, D);
}

torch::Tensor dense_idlv(
    torch::Tensor level_hvs_packed,
    torch::Tensor id_hvs_packed,
    torch::Tensor raw_data,
    const int N, const int Q, const int F, const int D) {
    
    CHECK_INPUT(level_hvs_packed);
    CHECK_INPUT(id_hvs_packed);
    CHECK_INPUT(raw_data);

    return dense_idlv_cuda(level_hvs_packed, id_hvs_packed, raw_data, N, Q, F, D);
}

torch::Tensor hd_enc_lvid(
    torch::Tensor level_hvs,
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor feature_values,
    torch::Tensor csr_info,
    const int N, const int Q, const int D) {
    
    CHECK_INPUT(feature_indices);
    CHECK_INPUT(feature_values);
    CHECK_INPUT(csr_info);
    CHECK_INPUT(level_hvs);
    CHECK_INPUT(id_hvs);

    return hd_enc_lvid_cuda(level_hvs, id_hvs, feature_indices, feature_values, csr_info, N, Q, D);
}

torch::Tensor hd_enc_id(
    torch::Tensor id_hvs,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D) {
    
    CHECK_INPUT(feature_indices);
    CHECK_INPUT(csr_info);
    CHECK_INPUT(id_hvs);

    return hd_enc_id_cuda(id_hvs, feature_indices, csr_info, N, D);
}

torch::Tensor hd_enc_id_packed(
    torch::Tensor id_hvs_packed,
    torch::Tensor feature_indices,
    torch::Tensor csr_info,
    const int N, const int D) {
    
    CHECK_INPUT(feature_indices);
    CHECK_INPUT(csr_info);
    CHECK_INPUT(id_hvs_packed);

    return hd_enc_id_packed_cuda(id_hvs_packed, feature_indices, csr_info, N, D);
}

torch::Tensor memory_hv_enc(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D) {
    
    CHECK_INPUT(node_hvs);
    CHECK_INPUT(idx);
    CHECK_INPUT(idx_ptr);

    return memory_hv_enc_cuda(node_hvs, idx, idx_ptr, N, D);
}

torch::Tensor memory_hv_lv2_enc(
    torch::Tensor node_hvs,
    torch::Tensor idx,
    torch::Tensor idx_ptr,
    const int N, const int D) {
    
    CHECK_INPUT(node_hvs);
    CHECK_INPUT(idx);
    CHECK_INPUT(idx_ptr);

    return memory_hv_lv2_enc_cuda(node_hvs, idx, idx_ptr, N, D);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lvid", &hd_enc_lvid, "HD LV-ID Encoding (CUDA)");
    m.def("packing", &packing, "Packing (CUDA)");
    m.def("dense_idlv", &dense_idlv, "dense_idlv (CUDA)");
    m.def("id_enc", &hd_enc_id, "HD ID Encoding (CUDA)");
    m.def("id_packed_enc", &hd_enc_id_packed, "HD ID (Packed) Encoding (CUDA)");
    m.def("memory_hv_enc", &memory_hv_enc, "Memory HV Build (CUDA)");
    m.def("memory_hv_lv2_enc", &memory_hv_lv2_enc, "Memory HV Build (CUDA)");
}
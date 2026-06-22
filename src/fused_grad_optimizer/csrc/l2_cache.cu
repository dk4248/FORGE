/*
 * L2 Cache Pinning Extension
 *
 * Uses cudaStreamSetAttribute with cudaStreamAttributeAccessPolicyWindow
 * to hint the GPU that certain memory regions should persist in L2 cache.
 *
 * This is Level 2 of the memory hierarchy optimization: activation tensors
 * that are re-read by every tile of the fused kernel benefit from staying
 * in L2 (~200 cycle latency) instead of being evicted back to HBM (~400 cycles).
 *
 * API: CUDA 11.0+ (cudaAccessPolicyWindow)
 * Hardware: SM 80+ (Ampere, Hopper, Blackwell)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Check CUDA errors with file/line context
#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        TORCH_CHECK(err == cudaSuccess,                                       \
                    "CUDA error in " #call ": ", cudaGetErrorString(err));     \
    } while (0)


int64_t get_persisting_l2_max_size() {
    /*
     * Returns the maximum number of bytes that can be set to persist in L2.
     * This is a hardware property — typically a fraction of the total L2.
     * RTX PRO 6000 Blackwell: 128 MB total L2, persisting max varies.
     */
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int persisting_max = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &persisting_max,
        cudaDevAttrMaxPersistingL2CacheSize,
        device
    ));
    return static_cast<int64_t>(persisting_max);
}


int64_t get_l2_cache_size() {
    /* Returns total L2 cache size in bytes. */
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int l2_size = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &l2_size,
        cudaDevAttrL2CacheSize,
        device
    ));
    return static_cast<int64_t>(l2_size);
}


void set_persisting_l2_max(int64_t max_bytes) {
    /*
     * Set the maximum L2 cache set-aside for persisting accesses.
     * Must be <= get_persisting_l2_max_size().
     * Call this once at startup to reserve L2 space for our kernel.
     */
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int hw_max = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &hw_max,
        cudaDevAttrMaxPersistingL2CacheSize,
        device
    ));

    int actual = std::min(static_cast<int>(max_bytes), hw_max);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, actual));
}


void pin_l2(torch::Tensor tensor, double hit_ratio) {
    /*
     * Set L2 persistence hint for a tensor on the current CUDA stream.
     *
     * After this call, loads from this memory region on this stream will
     * hint the L2 cache controller to keep the data persistent (don't evict).
     *
     * Args:
     *   tensor: The CUDA tensor to pin in L2.
     *   hit_ratio: Fraction of the tensor to mark as persisting [0.0, 1.0].
     *              1.0 = try to keep the entire tensor in L2.
     *              Use < 1.0 if the tensor is larger than persisting L2 budget.
     */
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(hit_ratio >= 0.0 && hit_ratio <= 1.0,
                "hit_ratio must be in [0.0, 1.0]");

    // Get persisting L2 max
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int persisting_max = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &persisting_max,
        cudaDevAttrMaxPersistingL2CacheSize,
        device
    ));

    size_t tensor_bytes = tensor.nbytes();

    // Clamp to persisting max
    size_t num_bytes = std::min(tensor_bytes, static_cast<size_t>(persisting_max));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Set the access policy window
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr = tensor.data_ptr();
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio = static_cast<float>(hit_ratio);
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    ));
}


void unpin_l2() {
    /*
     * Reset L2 persistence hints on the current CUDA stream.
     * Call after the kernel that benefits from pinning has completed.
     */
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Reset the window (set num_bytes to 0)
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.num_bytes = 0;

    CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    ));

    // Also reset the persisting L2 cache lines
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());
}


std::tuple<int64_t, int64_t> get_l2_info() {
    /*
     * Returns (total_l2_bytes, max_persisting_bytes).
     * Useful for deciding what to pin and logging.
     */
    return std::make_tuple(get_l2_cache_size(), get_persisting_l2_max_size());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "L2 Cache Pinning Extension for fused grad+optimizer kernels";
    m.def("pin_l2", &pin_l2,
          "Set L2 persistence hint for a CUDA tensor",
          py::arg("tensor"), py::arg("hit_ratio") = 1.0);
    m.def("unpin_l2", &unpin_l2,
          "Reset L2 persistence hints on current stream");
    m.def("get_l2_info", &get_l2_info,
          "Returns (total_l2_bytes, max_persisting_bytes)");
    m.def("get_persisting_l2_max_size", &get_persisting_l2_max_size,
          "Max bytes that can persist in L2");
    m.def("get_l2_cache_size", &get_l2_cache_size,
          "Total L2 cache size in bytes");
    m.def("set_persisting_l2_max", &set_persisting_l2_max,
          "Set max L2 set-aside for persisting accesses",
          py::arg("max_bytes"));
}

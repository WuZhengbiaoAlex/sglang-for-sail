#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace sglang { namespace incremental_eplb {
static constexpr int kWarp = 32;
static constexpr int kWarpsPerBlock = 4;
static constexpr int kMaxRanks = 64;
static constexpr int kMaxExperts = 1024;
// One warp per layer: physical-slot loads -> logical-expert loads.
//   gathered [R, L, npe]  tokens received by each rank's local physical slots
//   p2l      [L, P]       the placement those counts were measured under
//   out      [L, E]        global load per logical expert
//
// Every rank reduces the same collected physical counts through the placement,
// producing identical logical loads without another collective.
__global__ void phys_to_logical_kernel(const int* __restrict__ gathered,
                                       const int* __restrict__ p2l,
                                       float* __restrict__ out,
                                       int L, int E, int P, int R) {
  extern __shared__ char smem_raw[];
  const int warp_in_block = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int layer = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (layer >= L) return;

  const int npe = P / R;
  int* acc = reinterpret_cast<int*>(smem_raw) + (size_t)warp_in_block * E;

  for (int e = lane; e < E; e += kWarp) acc[e] = 0;
  __syncwarp();

  const int* my_p2l = p2l + (size_t)layer * P;
  for (int p = lane; p < P; p += kWarp) {
    const int e = my_p2l[p];
    if (e < 0 || e >= E) continue;
    const int r = p / npe, i = p % npe;
    const int n = gathered[((size_t)r * L + layer) * npe + i];
    if (n) atomicAdd(&acc[e], n);
  }
  __syncwarp();

  float* my_out = out + (size_t)layer * E;
  for (int e = lane; e < E; e += kWarp) my_out[e] = (float)acc[e];
}

void phys_to_logical(at::Tensor gathered, at::Tensor p2l, at::Tensor out,
                     int64_t num_ranks) {
  TORCH_CHECK(gathered.is_cuda() && gathered.scalar_type() == at::kInt &&
                  gathered.dim() == 3,
              "gathered must be a 3-D int32 CUDA tensor [ranks, layers, local]");
  TORCH_CHECK(p2l.is_cuda() && p2l.scalar_type() == at::kInt && p2l.dim() == 2);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 2);
  const int L = p2l.size(0), P = p2l.size(1), E = out.size(1);
  const int R = (int)num_ranks;
  TORCH_CHECK(R > 0 && R <= kMaxRanks, "num_ranks must be in (0, 64]");
  TORCH_CHECK(
      gathered.is_contiguous() && p2l.is_contiguous() && out.is_contiguous(),
      "gathered, p2l, and out must be contiguous");
  TORCH_CHECK(
      gathered.get_device() == p2l.get_device() &&
          p2l.get_device() == out.get_device(),
      "gathered, p2l, and out must be on the same device");
  TORCH_CHECK(P % R == 0, "physical experts must divide by num_ranks");
  TORCH_CHECK(gathered.size(0) == R && gathered.size(1) == L &&
                  gathered.size(2) == P / R,
              "gathered shape must be [num_ranks, layers, num_physical/num_ranks]");
  TORCH_CHECK(out.size(0) == L && E > 0 && E <= kMaxExperts);

  const int smem = (int)((size_t)E * sizeof(int) * kWarpsPerBlock);
  const int blocks = (L + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        phys_to_logical_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  phys_to_logical_kernel<<<blocks, kWarp * kWarpsPerBlock, smem, stream>>>(
      gathered.data_ptr<int>(), p2l.data_ptr<int>(), out.data_ptr<float>(),
      L, E, P, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef SGLANG_INCREMENTAL_PEER_MEMORY
// Defined in peer_gather.cu.
void publish_counts(at::Tensor region, at::Tensor phys, int64_t step);
void peer_gather(at::Tensor peer_ptrs, at::Tensor gathered,
                 int64_t plane_ints, int64_t num_ranks, int64_t step);
void sync_copy_generation(at::Tensor peer_ptrs, at::Tensor local_region,
                          int64_t completion_offset, int64_t num_ranks,
                          int64_t generation);
#endif
}  // namespace incremental_eplb
}  // namespace sglang

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("phys_to_logical", &sglang::incremental_eplb::phys_to_logical);
#ifdef SGLANG_INCREMENTAL_PEER_MEMORY
  m.def("publish_counts", &sglang::incremental_eplb::publish_counts);
  m.def("peer_gather", &sglang::incremental_eplb::peer_gather);
  m.def("sync_copy_generation", &sglang::incremental_eplb::sync_copy_generation);
#endif
}

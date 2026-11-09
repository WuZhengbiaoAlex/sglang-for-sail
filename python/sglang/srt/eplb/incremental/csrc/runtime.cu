// Copyright 2026 SGLang Team. Licensed under the Apache License, Version 2.0.
// Multi-owner split routing for incremental EPLB.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cassert>
#include <climits>
#include <torch/extension.h>

namespace {
constexpr int MAX_R = 64;
template <typename Id, typename Count>
__global__ void route_primary(const Id *ids, const int *owners,
                              const Count *valid, int64_t *out, int N, int K,
                              int E, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  int64_t e = ids[i];
  out[i] = (i / K < valid[0] && e >= 0 && e < E) ? owners[width * e] : -1;
}

template <typename Id, typename Count>
__global__ void route_split(const Id *ids, const int *owners,
                            const Count *valid, int64_t *out, int N, int K,
                            int rank, int width) {
  int e = blockIdx.x, count_owners = 0;
  while (count_owners < width && owners[width * e + count_owners] >= 0)
    ++count_owners;
  if (count_owners < 2)
    return;
  int lane = threadIdx.x;
  int count = 0,
      end = min(static_cast<int64_t>(N), static_cast<int64_t>(valid[0]) * K);
  for (int base = 0; base < end; base += 32) {
    int i = base + lane;
    bool match = i < end && ids[i] == e;
    unsigned mask = __ballot_sync(0xffffffff, match);
    unsigned lower = lane == 0 ? 0u : (0xffffffffu >> (32 - lane));
    int ordinal = count + __popc(mask & lower);
    if (match)
      out[i] = owners[width * e + (ordinal + rank + e) % count_owners];
    count += __popc(mask);
  }
}

template <typename Id, typename Count>
void route_fallback_launch(torch::Tensor ids, torch::Tensor owners,
                           torch::Tensor valid, torch::Tensor out, int rank) {
  int N = ids.numel(), K = ids.size(1), E = owners.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N > 0) {
    route_primary<Id, Count><<<(N + 255) / 256, 256, 0, stream>>>(
        ids.data_ptr<Id>(), owners.data_ptr<int>(), valid.data_ptr<Count>(),
        out.data_ptr<int64_t>(), N, K, E, owners.size(1));
    route_split<Id, Count><<<E, 32, 0, stream>>>(
        ids.data_ptr<Id>(), owners.data_ptr<int>(), valid.data_ptr<Count>(),
        out.data_ptr<int64_t>(), N, K, rank, owners.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <typename Id, typename Count, int MAX_N>
__global__ void route_prefix_kernel(const Id *ids, const int *owners,
                                    const Count *valid, int64_t *out, int N,
                                    int K, int E, int W, int rank) {
  // Prefix counts preserve occurrence order across all warps of this expert.
  constexpr int T = 256;
  int primary = (N + T - 1) / T;
  if (blockIdx.x < primary) {
    int i = blockIdx.x * T + threadIdx.x;
    if (i >= N)
      return;
    int64_t e = ids[i];
    if (i / K >= valid[0] || e < 0 || e >= E)
      out[i] = -1;
    else if (W < 2 || owners[e * W + 1] < 0)
      out[i] = owners[e * W];
    return;
  }
  int e = blockIdx.x - primary;
  if (W < 2 || owners[e * W + 1] < 0)
    return;
  int n = 0;
  while (n < W && owners[e * W + n] >= 0)
    ++n;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  __shared__ int prefix[MAX_N / 32];
  unsigned masks[MAX_N / T];
  bool matches[MAX_N / T];
  int end = min(static_cast<int64_t>(N), static_cast<int64_t>(valid[0]) * K);
#pragma unroll
  for (int j = 0; j < MAX_N / T; ++j) {
    int i = j * T + threadIdx.x;
    bool match = i < end && ids[i] == e;
    unsigned mask = __ballot_sync(0xffffffff, match);
    masks[j] = mask;
    matches[j] = match;
    if (lane == 0)
      prefix[j * (T / 32) + warp] = __popc(mask);
  }
  __syncthreads();
  if (warp == 0) {
    int carry = 0;
#pragma unroll
    for (int chunk = 0; chunk < MAX_N / 1024; ++chunk) {
      int val = prefix[chunk * 32 + lane], original = val;
#pragma unroll
      for (int delta = 1; delta < 32; delta *= 2) {
        int other = __shfl_up_sync(0xffffffff, val, delta);
        if (lane >= delta)
          val += other;
      }
      prefix[chunk * 32 + lane] = carry + val - original;
      carry += __shfl_sync(0xffffffff, val, 31);
    }
  }
  __syncthreads();
  unsigned lower = lane == 0 ? 0u : (0xffffffffu >> (32 - lane));
#pragma unroll
  for (int j = 0; j < MAX_N / T; ++j) {
    int i = j * T + threadIdx.x;
    if (matches[j]) {
      int ordinal = prefix[j * (T / 32) + warp] + __popc(masks[j] & lower);
      out[i] = owners[e * W + (ordinal + rank + e) % n];
    }
  }
}
template <typename Id, typename Count>
void route_launch(torch::Tensor ids, torch::Tensor owners, torch::Tensor valid,
                  torch::Tensor out, int rank) {
  int N = ids.numel(), K = ids.size(1), E = owners.size(0), W = owners.size(1);
  if (N > 4096) {
    route_fallback_launch<Id, Count>(ids, owners, valid, out, rank);
    return;
  }
  if (N > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    if (N <= 1024)
      route_prefix_kernel<Id, Count, 1024>
          <<<E + (N + 255) / 256, 256, 0, stream>>>(
              ids.data_ptr<Id>(), owners.data_ptr<int>(),
              valid.data_ptr<Count>(), out.data_ptr<int64_t>(), N, K, E, W,
              rank);
    else if (N <= 2048)
      route_prefix_kernel<Id, Count, 2048>
          <<<E + (N + 255) / 256, 256, 0, stream>>>(
              ids.data_ptr<Id>(), owners.data_ptr<int>(),
              valid.data_ptr<Count>(), out.data_ptr<int64_t>(), N, K, E, W,
              rank);
    else
      route_prefix_kernel<Id, Count, 4096>
          <<<E + (N + 255) / 256, 256, 0, stream>>>(
              ids.data_ptr<Id>(), owners.data_ptr<int>(),
              valid.data_ptr<Count>(), out.data_ptr<int64_t>(), N, K, E, W,
              rank);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor unified_route(torch::Tensor ids, torch::Tensor owners,
                            torch::Tensor valid, int64_t rank) {
  TORCH_CHECK(ids.is_cuda() && ids.device() == owners.device() &&
                  ids.device() == valid.device(),
              "Device mismatch");
  TORCH_CHECK(ids.dim() == 2 && ids.size(1) > 0 && owners.dim() == 2 &&
                  owners.size(1) > 0 && owners.size(1) <= MAX_R &&
                  valid.numel() == 1,
              "Invalid routing shapes");
  TORCH_CHECK(owners.scalar_type() == torch::kInt32 &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64) &&
                  (valid.scalar_type() == torch::kInt32 ||
                   valid.scalar_type() == torch::kInt64),
              "Invalid routing dtypes");
  TORCH_CHECK(ids.is_contiguous() && owners.is_contiguous() &&
                  valid.is_contiguous() && rank >= 0,
              "Invalid routing inputs");
  c10::cuda::CUDAGuard guard(ids.device());
  auto out = torch::empty(ids.sizes(), ids.options().dtype(torch::kInt64));
  if (ids.scalar_type() == torch::kInt32) {
    if (valid.scalar_type() == torch::kInt32)
      route_launch<int, int>(ids, owners, valid, out, rank);
    else
      route_launch<int, int64_t>(ids, owners, valid, out, rank);
  } else {
    if (valid.scalar_type() == torch::kInt32)
      route_launch<int64_t, int>(ids, owners, valid, out, rank);
    else
      route_launch<int64_t, int64_t>(ids, owners, valid, out, rank);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route", &unified_route,
        "Deterministic local-quota multi-owner routing");
}

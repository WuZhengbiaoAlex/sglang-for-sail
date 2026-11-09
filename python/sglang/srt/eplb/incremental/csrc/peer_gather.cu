// Copyright 2023-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// One-sided collection of the per-rank expert counts used by IncrementalEPLB.
//
// Each rank publishes one count plane into symmetric memory and then pulls the
// same generation from every peer. There is no process-group collective on the
// decode hot path. Correctness depends on two rules enforced here:
//
//   1. A slot's sequence number is published only after every payload writer
//      has completed a system-scope fence.
//   2. A reader accepts only the exact requested generation. A ring slot that
//      has already been reused is never mistaken for the requested data.
//
// Publication deliberately uses one block. __syncthreads() is block-local, so
// a multi-block publisher cannot safely advertise completion with one flag.
// Gathering likewise uses one block per source rank, making the sequence check
// cover every byte copied for that source.

#include <limits>
#include <cassert>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace sglang {
namespace incremental_eplb {

static constexpr int kRingSlots = 4;
static constexpr int kSlotHeaderInts = 32;  // 128-byte payload alignment

// A diverged or missing peer must fail instead of spinning indefinitely.
__device__ void wait_generation(const volatile long long* sequence,
                               long long expected, unsigned long long timeout) {
  const unsigned long long start = clock64();
  for (;;) {
    const long long observed = *sequence;
    if (observed == expected) return;
    assert(observed < expected && "Incremental EPLB peer generation was overwritten");
    assert(clock64() - start < timeout && "Incremental EPLB peer generation timed out");
    __nanosleep(64);
  }
}

unsigned long long generation_timeout_cycles(int device) {
  int clock_khz = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  TORCH_CHECK(clock_khz > 0, "GPU clock rate is required for peer wait timeout");
  return static_cast<unsigned long long>(clock_khz) * 1000 * 30;
}

__host__ __device__ inline int slot_stride_ints(int plane_ints) {
  return ((kSlotHeaderInts + plane_ints + 31) / 32) * 32;
}

__global__ void publish_counts_kernel(
    int* __restrict__ region,
    const int* __restrict__ phys,
    int plane_ints,
    long long step) {
  const int stride = slot_stride_ints(plane_ints);
  int* slot = region + (step % kRingSlots) * static_cast<long long>(stride);
  int* dst = slot + kSlotHeaderInts;

  for (int i = threadIdx.x; i < plane_ints; i += blockDim.x) {
    dst[i] = phys[i];
  }

  // Every writer releases its payload stores to system scope. The block
  // barrier then proves all releases completed before thread 0 advertises the
  // generation to peer GPUs.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile long long*>(slot) = step;
    __threadfence_system();
  }
}

__global__ void peer_gather_kernel(
    const int* const* __restrict__ peer_region,
    int* __restrict__ gathered,
    int plane_ints,
    int num_ranks,
    long long step, unsigned long long timeout) {
  const int rank = blockIdx.x;
  if (rank >= num_ranks) {
    return;
  }

  const int stride = slot_stride_ints(plane_ints);
  const int* slot =
      peer_region[rank] + (step % kRingSlots) * static_cast<long long>(stride);

  // Accept only the requested generation. Consuming a reused slot would make
  // placement diverge across ranks.
  if (threadIdx.x == 0) {
    wait_generation(reinterpret_cast<volatile const long long*>(slot), step, timeout);
  }
  __syncthreads();
  __threadfence_system();

  const volatile int* src = slot + kSlotHeaderInts;
  int* dst = gathered + static_cast<long long>(rank) * plane_ints;
  for (int i = threadIdx.x; i < plane_ints; i += blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void sync_copy_generation_kernel(
    const int* const* __restrict__ peer_region,
    int* __restrict__ local_region,
    int completion_offset,
    int num_ranks,
    long long generation, unsigned long long timeout) {
  if (threadIdx.x != 0) return;

  const int slot = static_cast<int>(generation % kRingSlots) * 2;
  volatile long long* local = reinterpret_cast<volatile long long*>(
      local_region + completion_offset + slot);
  __threadfence_system();
  *local = generation;
  __threadfence_system();

  for (int rank = 0; rank < num_ranks; ++rank) {
    const volatile long long* peer =
        reinterpret_cast<volatile const long long*>(
            peer_region[rank] + completion_offset + slot);
    wait_generation(peer, generation, timeout);
  }
}

void publish_counts(at::Tensor region, at::Tensor phys, int64_t step) {
  TORCH_CHECK(region.is_cuda() && phys.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(region.scalar_type() == at::kInt &&
                  phys.scalar_type() == at::kInt,
              "region and phys must be int32");
  TORCH_CHECK(region.is_contiguous() && phys.is_contiguous(),
              "region and phys must be contiguous");
  TORCH_CHECK(region.get_device() == phys.get_device(),
              "region and phys must be on the same device");
  TORCH_CHECK(step > 0, "step must be positive");
  TORCH_CHECK(phys.numel() > 0 &&
                  phys.numel() <= std::numeric_limits<int>::max(),
              "phys is empty or too large");

  const int plane = static_cast<int>(phys.numel());
  const int stride = slot_stride_ints(plane);
  TORCH_CHECK(region.numel() >= static_cast<int64_t>(kRingSlots) * stride,
              "published region too small: need ", kRingSlots * stride,
              " int32 values, got ", region.numel());

  publish_counts_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      region.data_ptr<int>(), phys.data_ptr<int>(), plane,
      static_cast<long long>(step));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_gather(at::Tensor peer_ptrs, at::Tensor gathered,
                 int64_t plane_ints, int64_t num_ranks, int64_t step) {
  TORCH_CHECK(peer_ptrs.is_cuda() && peer_ptrs.scalar_type() == at::kLong,
              "peer_ptrs must be an int64 CUDA tensor");
  TORCH_CHECK(gathered.is_cuda() && gathered.scalar_type() == at::kInt,
              "gathered must be an int32 CUDA tensor");
  TORCH_CHECK(peer_ptrs.is_contiguous() && gathered.is_contiguous(),
              "peer_ptrs and gathered must be contiguous");
  TORCH_CHECK(peer_ptrs.get_device() == gathered.get_device(),
              "peer_ptrs and gathered must be on the same device");
  TORCH_CHECK(plane_ints > 0 &&
                  plane_ints <= std::numeric_limits<int>::max(),
              "plane_ints must fit int32");
  TORCH_CHECK(num_ranks > 0 && num_ranks <= 64,
              "num_ranks must be in [1, 64]");
  TORCH_CHECK(peer_ptrs.numel() >= num_ranks,
              "peer_ptrs has fewer entries than num_ranks");
  TORCH_CHECK(gathered.numel() >= plane_ints * num_ranks,
              "gathered is too small");
  TORCH_CHECK(step > 0, "step must be positive");

  peer_gather_kernel<<<static_cast<int>(num_ranks), 32, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int* const*>(peer_ptrs.data_ptr<int64_t>()),
      gathered.data_ptr<int>(), static_cast<int>(plane_ints),
      static_cast<int>(num_ranks), static_cast<long long>(step),
      generation_timeout_cycles(gathered.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sync_copy_generation(at::Tensor peer_ptrs, at::Tensor local_region,
                          int64_t completion_offset, int64_t num_ranks,
                          int64_t generation) {
  TORCH_CHECK(peer_ptrs.is_cuda() && peer_ptrs.scalar_type() == at::kLong,
              "peer_ptrs must be an int64 CUDA tensor");
  TORCH_CHECK(local_region.is_cuda() &&
                  local_region.scalar_type() == at::kInt,
              "local_region must be an int32 CUDA tensor");
  TORCH_CHECK(peer_ptrs.is_contiguous() && local_region.is_contiguous(),
              "peer_ptrs and local_region must be contiguous");
  TORCH_CHECK(peer_ptrs.get_device() == local_region.get_device(),
              "peer_ptrs and local_region must be on the same device");
  TORCH_CHECK(num_ranks > 1 && num_ranks <= 64,
              "num_ranks must be in [2, 64]");
  TORCH_CHECK(peer_ptrs.numel() >= num_ranks,
              "peer_ptrs has fewer entries than num_ranks");
  TORCH_CHECK(completion_offset >= 0 && completion_offset % 2 == 0,
              "completion_offset must be a non-negative int64-aligned offset");
  TORCH_CHECK(local_region.numel() >= completion_offset + kRingSlots * 2,
              "local_region is too small for the completion ring");
  TORCH_CHECK(generation > 0, "generation must be positive");

  sync_copy_generation_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int* const*>(peer_ptrs.data_ptr<int64_t>()),
      local_region.data_ptr<int>(), static_cast<int>(completion_offset),
      static_cast<int>(num_ranks), static_cast<long long>(generation),
      generation_timeout_cycles(local_region.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace incremental_eplb
}  // namespace sglang

// Exact two-stage TopK for the small query batch left after attention-TP splitting.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

namespace sglang {
namespace ppu_topk_split {
namespace impl = device::topk;
using Register2 = impl::TopKRegister<2>;
using Register4 = impl::TopKRegister<4>;
using Streaming = impl::TopKStreaming;
constexpr uint32_t kTopK = 2048;

struct ValueProblem : impl::TopKProblem {
  float* values;

  SGL_DEVICE void emit(uint32_t pos, uint32_t idx) const {
    emit(pos, idx, in[idx]);
  }
  SGL_DEVICE void emit(uint32_t pos, uint32_t idx, float value) const {
    TopKProblem::emit(pos, idx);
    values[pos] = value;
  }
};

struct Params {
  const float* scores;
  const int32_t* lengths;
  const int32_t* pages;
  int32_t* output;
  float* values;
  const int32_t* indirect;
  int32_t* candidate_lengths;
  uint32_t columns, splits, chunk;
  int64_t score_stride, page_stride;
};

template <bool Partial, bool Map>
__global__ __launch_bounds__(1024, 2) void select(const __grid_constant__ Params p) {
  device::enable_smem_spilling();
  using Workspace = impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem>;
  __shared__ __align__(16) uint8_t workspace[sizeof(Workspace)];
  __shared__ int32_t selected[kTopK];
  const uint32_t part = blockIdx.x % p.splits;
  const uint32_t row = blockIdx.x / p.splits;
  const uint32_t offset = part * p.chunk;
  const uint32_t total = p.lengths[row];
  const uint32_t length = total > offset ? min(total - offset, p.chunk) : 0;
  auto* result = p.output + static_cast<int64_t>(blockIdx.x) * kTopK;
  if constexpr (Partial) {
    // Chunks >= K make valid candidates a contiguous prefix, including short
    // and empty rows. Refresh on every execution, including graph replay.
    if (part == 0 && threadIdx.x == 0) {
      p.candidate_lengths[row] = min(total / p.chunk, p.splits) * kTopK + min(total % p.chunk, kTopK);
    }
  }
  if (length <= kTopK) {
    for (uint32_t j = threadIdx.x; j < kTopK; j += blockDim.x) {
      int32_t idx = -1;
      if (j < length) {
        if constexpr (Partial)
          idx = offset + j;
        else
          idx = p.indirect[row * p.columns + j];
      }
      if constexpr (Map) {
        result[j] = idx < 0 ? -1 : p.pages[row * p.page_stride + idx / 64] * 64 + idx % 64;
      } else {
        result[j] = idx;
      }
      if constexpr (Partial) {
        p.values[static_cast<int64_t>(blockIdx.x) * kTopK + j] =
            j < length ? p.scores[row * p.score_stride + offset + j] : -std::numeric_limits<float>::infinity();
      }
    }
    return;
  }
  impl::TopKProblem problem{
      .in = p.scores + row * p.score_stride + offset,
      .out = Partial ? result : selected,
      .topk = kTopK,
      .seq_len = length,
      .bias = static_cast<int32_t>(offset),
  };
  if constexpr (Partial) {
    ValueProblem value_problem{problem, p.values + static_cast<int64_t>(blockIdx.x) * kTopK};
    if (length <= Register2::kMaxSeqLen)
      Register2::forward<false>(value_problem, workspace);
    else
      Register4::forward<false>(value_problem, workspace);
  } else {
    if (length <= Register2::kMaxSeqLen)
      Register2::forward<false>(problem, workspace);
    else if (length <= Register4::kMaxSeqLen)
      Register4::forward<false>(problem, workspace);
    else
      Streaming::forward<false>(problem, workspace);
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < kTopK; j += blockDim.x) {
      const auto idx = p.indirect[row * p.columns + selected[j]];
      if constexpr (Map)
        result[j] = p.pages[row * p.page_stride + idx / 64] * 64 + idx % 64;
      else
        result[j] = idx;
    }
  }
}
}  // namespace ppu_topk_split

struct PPUTopKSplit {
  static void partial(
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView values,
      tvm::ffi::TensorView candidate_lengths,
      uint32_t splits) {
    using namespace host;
    using namespace ppu_topk_split;
    auto rows = SymbolicSize{"rows"};
    auto cols = SymbolicSize{"columns"};
    auto stride = SymbolicSize{"stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({rows, cols}).with_strides({stride, 1}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows, splits * kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({rows, splits * kTopK}).with_dtype<float>().with_device(device).verify(values);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(candidate_lengths);
    RuntimeCheck(splits > 1, "split TopK requires multiple chunks");
    const uint32_t chunk = ((cols.unwrap() + splits * 4 - 1) / (splits * 4)) * 4;
    RuntimeCheck(chunk >= kTopK && chunk <= Register4::kMaxSeqLen, "invalid TopK chunk size");
    RuntimeCheck(stride.unwrap() % 4 == 0, "TopK rows must be float4 aligned");
    const Params params{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .pages = nullptr,
        .output = static_cast<int32_t*>(indices.data_ptr()),
        .values = static_cast<float*>(values.data_ptr()),
        .indirect = nullptr,
        .candidate_lengths = static_cast<int32_t*>(candidate_lengths.data_ptr()),
        .columns = static_cast<uint32_t>(cols.unwrap()),
        .splits = splits,
        .chunk = chunk,
        .score_stride = stride.unwrap(),
        .page_stride = 0,
    };
    LaunchKernel(rows.unwrap() * splits, 1024, device.unwrap()).launch(select<true, false>, params);
  }

  static void merge(
      tvm::ffi::TensorView values,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView indices,
      tvm::ffi::Optional<tvm::ffi::TensorView> pages,
      tvm::ffi::TensorView output) {
    using namespace host;
    using namespace ppu_topk_split;
    auto rows = SymbolicSize{"rows"};
    auto cols = SymbolicSize{"columns"};
    auto stride = SymbolicSize{"stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({rows, cols}).with_strides({stride, 1}).with_dtype<float>().with_device(device).verify(values);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows, cols}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({rows, kTopK}).with_dtype<int32_t>().with_device(device).verify(output);
    if (pages.has_value()) {
      TensorMatcher({rows, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device).verify(pages.value());
    }
    const Params params{
        .scores = static_cast<const float*>(values.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .pages = pages.has_value() ? static_cast<const int32_t*>(pages.value().data_ptr()) : nullptr,
        .output = static_cast<int32_t*>(output.data_ptr()),
        .values = nullptr,
        .indirect = static_cast<const int32_t*>(indices.data_ptr()),
        .candidate_lengths = nullptr,
        .columns = static_cast<uint32_t>(cols.unwrap()),
        .splits = 1,
        .chunk = static_cast<uint32_t>(cols.unwrap()),
        .score_stride = stride.unwrap(),
        .page_stride = pages.has_value() ? pages.value().stride(0) : 0,
    };
    if (pages.has_value())
      LaunchKernel(rows.unwrap(), 1024, device.unwrap()).launch(select<false, true>, params);
    else
      LaunchKernel(rows.unwrap(), 1024, device.unwrap()).launch(select<false, false>, params);
  }
};
}  // namespace sglang

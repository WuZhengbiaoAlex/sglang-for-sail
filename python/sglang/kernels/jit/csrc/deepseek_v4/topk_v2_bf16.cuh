#pragma once

#include "topk_v2.cuh"

#include <type_traits>

namespace sglang::device::topk {

// All NaNs rank above +inf. Signed zeros may have distinct keys because their
// scores are equal and paged top-k is unordered.
SGL_DEVICE uint32_t bf16_ordered_key(uint32_t bits) {
  const auto negative = static_cast<uint32_t>(static_cast<int32_t>(bits << 16) >> 31);
  const auto ordered = bits ^ ((negative & 0xffffu) | 0x8000u);
  return (bits & 0x7fffu) > 0x7f80u ? 0xffffu : ordered;
}

template <typename To, typename From>
SGL_DEVICE To bf16_topk_bitcast(const From& from) {
  static_assert(sizeof(To) == sizeof(From));
  return reinterpret_cast<const To&>(from);
}

SGL_DEVICE bf16x2_t bf16_cutoff_pair(uint32_t key) {
  const auto raw = (key & 0x8000u) ? (key ^ 0x8000u) : (key ^ 0xffffu);
  return bf16_topk_bitcast<bf16x2_t>(raw | (raw << 16));
}

template <uint32_t kLocalVecs>
struct BF16PagedTopKImpl : TopKConfig {
  using Problem = TopKProblem;
  static constexpr uint32_t kVecSize = 8;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
  static constexpr bool kRegisterResident = kLocalVecs != 0;
  static constexpr uint32_t kFineBits = 3;
  static constexpr uint32_t kFineSize = 1u << kFineBits;
  static constexpr uint32_t kFineMask = kFineSize - 1;
  static constexpr uint32_t kHistSize = 1u << (16 - kFineBits);
  static constexpr uint32_t kHistItems = kHistSize / kBlockSize;
  static constexpr uint32_t kCapacity = kRegisterResident ? 4096 : 8192;
  static constexpr uint32_t kMaxPackedRowLength = 1u << (32 - kFineBits);
  using Vec = AlignedVector<uint16_t, kVecSize>;

  struct Candidate {
    uint32_t index;
    uint32_t fine;
  };
  using CachedCandidate = std::conditional_t<kRegisterResident, Candidate, uint32_t>;
  static_assert(sizeof(CachedCandidate) * kCapacity == sizeof(uint32_t) * kHistSize);

  struct Smem {
    // Coarse counts are dead once the cutoff bucket has been published.
    union {
      alignas(16) uint32_t histogram[kHistSize];
      CachedCandidate candidates[kCapacity];
    };
    uint32_t warp_sum[kNumWarps];
    uint32_t fine_histogram[kFineSize];
    uint32_t low_key;
    uint32_t bucket_size;
    uint32_t above_bucket;
    uint32_t above_written;
    uint32_t cached;
    uint32_t pivot;
    uint32_t above_pivot;
    uint32_t fine_written;
    uint32_t equal_written;
  };

  SGL_DEVICE static uint32_t ordered_key(uint32_t element) {
    if constexpr (kRegisterResident) {
      return element;  // Converted once as the scores enter the registers.
    } else {
      return bf16_ordered_key(element);
    }
  }

  SGL_DEVICE static CachedCandidate encode_candidate(uint32_t index, uint32_t fine) {
    if constexpr (kRegisterResident) {
      return {index, fine};
    } else {
      return (index << kFineBits) | fine;
    }
  }

  SGL_DEVICE static Candidate decode_candidate(CachedCandidate cached) {
    if constexpr (kRegisterResident) {
      return cached;
    } else {
      return {cached >> kFineBits, cached & kFineMask};
    }
  }

  SGL_DEVICE static void find_coarse(const Problem& problem, Smem& smem) {
    const auto tx = threadIdx.x;
    const auto lane = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    uint32_t prefix[kHistItems + 1] = {};
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      prefix[i + 1] = prefix[i] + smem.histogram[tx * kHistItems + i];
    }
    const auto inclusive = warp::inclusive_sum(prefix[kHistItems], lane);
    if (lane == kWarpSize - 1) smem.warp_sum[warp_id] = inclusive;
    __syncthreads();

    const auto previous_warps = warp::reduce_sum(lane < warp_id ? smem.warp_sum[lane] : 0u);
    const auto local_start = previous_warps + inclusive - prefix[kHistItems];
    const auto remaining = problem.seq_len - problem.topk - local_start;
    if (remaining < prefix[kHistItems]) {
#pragma unroll
      for (uint32_t i = 0; i < kHistItems; ++i) {
        if (prefix[i] <= remaining && remaining < prefix[i + 1]) {
          smem.low_key = (tx * kHistItems + i) << kFineBits;
          smem.bucket_size = prefix[i + 1] - prefix[i];
          smem.above_bucket = problem.seq_len - local_start - prefix[i + 1];
        }
      }
    }
    __syncthreads();
  }

  template <bool kPairs, bool kFull, typename Emit>
  SGL_DEVICE static void visit(
      const Vec& vec, uint32_t base, uint32_t length, uint32_t low, bf16x2_t boundary, const Emit& emit) {
    if constexpr (kPairs && kFull) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; j += 2) {
        const auto packed = static_cast<uint32_t>(vec[j]) | (static_cast<uint32_t>(vec[j + 1]) << 16);
        const auto hit = __hgeu2_mask(bf16_topk_bitcast<bf16x2_t>(packed), boundary);
        if (hit & 1u) emit(base + j, bf16_ordered_key(vec[j]));
        if (hit & 0x10000u) emit(base + j + 1, bf16_ordered_key(vec[j + 1]));
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (!kFull && base + j >= length) continue;
        const auto key = ordered_key(vec[j]);
        if (key >= low) emit(base + j, key);
      }
    }
  }

  template <typename Read, typename Emit>
  SGL_DEVICE static void visit_inputs(const Problem& problem, const Read& read, uint32_t low, const Emit& emit) {
    if constexpr (!kRegisterResident) {
      // Normal thresholds and infinities can use unordered BF16 pair compares.
      // Raw-key comparison handles zeros, subnormals, and the NaN cutoff.
      if ((low >= 0x007fu && low <= 0x7f7fu) || (low >= 0x8080u && low <= 0xff80u)) {
        const auto boundary = bf16_cutoff_pair(low);
        read([&]<bool kFull>(const Vec& vec, uint32_t base) {
          visit<true, kFull>(vec, base, problem.seq_len, low, boundary, emit);
        });
        return;
      }
    }
    read([&]<bool kFull>(const Vec& vec, uint32_t base) {
      visit<false, kFull>(vec, base, problem.seq_len, low, bf16x2_t{}, emit);
    });
  }

  template <bool kCache, typename Read>
  SGL_DEVICE static void collect(const Problem& problem, Smem& smem, const Read& read) {
    const auto low = smem.low_key;
    const auto high = low + kFineSize;
    visit_inputs(problem, read, low, [&](uint32_t index, uint32_t key) {
      if (key >= high) {
        problem.emit(atomicAdd(&smem.above_written, 1u), index);
      } else {
        const auto fine = key & kFineMask;
        if constexpr (kCache) {
          const auto slot = atomicAdd(&smem.cached, 1u);
          smem.candidates[slot] = encode_candidate(index, fine);
        }
        if constexpr (!kCache || kRegisterResident) atomicAdd(&smem.fine_histogram[fine], 1u);
      }
    });
  }

  template <typename Read>
  SGL_DEVICE static void select(const Problem& problem, Smem& smem, const Read& read) {
    const auto tx = threadIdx.x;
#pragma unroll
    for (uint32_t i = tx; i < kHistSize; i += kBlockSize) smem.histogram[i] = 0;
    if (tx < kFineSize) smem.fine_histogram[tx] = 0;
    if (tx == 0) {
      smem.above_written = 0;
      smem.cached = 0;
      smem.fine_written = 0;
      smem.equal_written = 0;
    }
    __syncthreads();

    read([&]<bool kFull>(const Vec& vec, uint32_t base) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (kFull || base + j < problem.seq_len) {
          atomicAdd(&smem.histogram[ordered_key(vec[j]) >> kFineBits], 1u);
        }
      }
    });
    __syncthreads();
    find_coarse(problem, smem);

    const auto count = smem.bucket_size;
    const auto base = smem.above_bucket;
    const auto remaining = problem.topk - base;
    const bool fits = count <= kCapacity && (kRegisterResident || problem.seq_len <= kMaxPackedRowLength);
    if (fits) {
      collect<true>(problem, smem, read);
      if constexpr (!kRegisterResident) {
        __syncthreads();
        for (uint32_t i = tx; i < count; i += kBlockSize) {
          atomicAdd(&smem.fine_histogram[decode_candidate(smem.candidates[i]).fine], 1u);
        }
      }
    } else {
      // Count every entry in the threshold bucket before rescanning for indices.
      collect<false>(problem, smem, read);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t above = 0;
      for (uint32_t fine = kFineSize; fine-- > 0;) {
        const auto hits = smem.fine_histogram[fine];
        if (above + hits >= remaining) {
          smem.pivot = fine;
          smem.above_pivot = above;
          break;
        }
        above += hits;
      }
    }
    __syncthreads();

    const auto pivot = smem.pivot;
    const auto above_pivot = smem.above_pivot;
    const auto emit = [&](uint32_t index, uint32_t fine) {
      if (fine > pivot) {
        problem.emit(base + atomicAdd(&smem.fine_written, 1u), index);
      } else if (fine == pivot) {
        const auto slot = above_pivot + atomicAdd(&smem.equal_written, 1u);
        if (slot < remaining) problem.emit(base + slot, index);
      }
    };
    if (fits) {
      for (uint32_t i = tx; i < count; i += kBlockSize) {
        const auto candidate = decode_candidate(smem.candidates[i]);
        emit(candidate.index, candidate.fine);
      }
    } else {
      const auto low = smem.low_key;
      const auto high = low + kFineSize;
      visit_inputs(problem, read, low + pivot, [&](uint32_t index, uint32_t key) {
        if (key < high) emit(index, key & kFineMask);
      });
    }
  }

  SGL_DEVICE static void load_tail(Vec& vec, const uint16_t* input, uint32_t base, uint32_t length) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      vec[j] = base + j < length ? input[base + j] : 0u;
    }
  }

  SGL_DEVICE static void forward(const uint16_t* input, const Problem& problem, void* storage) {
    auto& smem = *static_cast<Smem*>(storage);
    const auto tx = threadIdx.x;
    const auto full = problem.seq_len / kVecSize;
    const auto total = div_ceil(problem.seq_len, kVecSize);
    if constexpr (kRegisterResident) {
      Vec vectors[kLocalVecs];
#pragma unroll
      for (uint32_t i = 0; i < kLocalVecs; ++i) {
        const auto vi = tx + i * kBlockSize;
        if (vi < total) {
          if (vi < full) vectors[i].load(input, vi);
          else load_tail(vectors[i], input, vi * kVecSize, problem.seq_len);
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            vectors[i][j] = static_cast<uint16_t>(bf16_ordered_key(vectors[i][j]));
          }
        }
      }
      select(problem, smem, [&](const auto& consume) {
#pragma unroll
        for (uint32_t i = 0; i < kLocalVecs; ++i) {
          const auto vi = tx + i * kBlockSize;
          if (vi < full) consume.template operator()<true>(vectors[i], vi * kVecSize);
          else if (vi < total) consume.template operator()<false>(vectors[i], vi * kVecSize);
        }
      });
    } else {
      select(problem, smem, [&](const auto& consume) {
        auto vi = tx;
        if (vi < full) {
          Vec next;
          next.load(input, vi);
#pragma unroll 1
          do {
            const auto current = next;
            const auto base = vi * kVecSize;
            vi += kBlockSize;
            if (vi < full) next.load(input, vi);
            consume.template operator()<true>(current, base);
          } while (vi < full);
        }
        if (vi == full && vi < total) {
          Vec tail;
          load_tail(tail, input, vi * kVecSize, problem.seq_len);
          consume.template operator()<false>(tail, vi * kVecSize);
        }
      });
    }
  }
};

}  // namespace sglang::device::topk

namespace sglang {

using BF16RegisterShort = impl::BF16PagedTopKImpl<1>;
using BF16RegisterLong = impl::BF16PagedTopKImpl<2>;
using BF16Streaming = impl::BF16PagedTopKImpl<0>;

struct BF16PagedParams {
  const uint16_t* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
};

template <int kLevel>
TOPK_KERNEL void bf16_paged_topk_kernel(const __grid_constant__ BF16PagedParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<BF16RegisterShort::Smem, BF16RegisterLong::Smem, BF16Streaming::Smem> smem;
  __shared__ int32_t selected[kMaxTopK];

  const auto row = blockIdx.x;
  const auto* input = params.scores + row * params.score_stride;
  auto* out = params.page_indices + row * static_cast<int64_t>(params.topk);
  const auto transform = PageTransform{params.page_table + row * params.page_table_stride, params.page_bits, nullptr};
  auto problem = TopKProblem{
      .in = nullptr,  // BF16 input is passed separately.
      .out = out,
      .topk = params.topk,
      .seq_len = static_cast<uint32_t>(params.seq_lens[row]),
  };
  if (problem.seq_len <= problem.topk) {
    return trivial_transform<false, TopKMode::PAGE_TABLE>(problem, transform);
  }

  problem.out = selected;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= BF16RegisterShort::kMaxSeqLen);
    BF16RegisterShort::forward(input, problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= BF16RegisterLong::kMaxSeqLen);
    BF16RegisterLong::forward(input, problem, &smem);
  } else if (problem.seq_len <= BF16RegisterLong::kMaxSeqLen) {
    BF16RegisterLong::forward(input, problem, &smem);
  } else {
    BF16Streaming::forward(input, problem, &smem);
  }
  __syncthreads();
  paged_transform<TopKMode::PAGE_TABLE>(problem, out, transform);
}

struct BF16PagedTopKKernel {
  static void transform_paged(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      uint32_t page_size) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(seq_lens);
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device).verify(page_table);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(page_indices);
    RuntimeCheck(std::has_single_bit(page_size), "page_size must be a power of 2");
    RuntimeCheck(B.unwrap() <= 1 || S.unwrap() % BF16Streaming::kVecSize == 0,
                 "BF16 score_stride must be a multiple of 8");
    RuntimeCheck(reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0, "BF16 scores must be 16-byte aligned");
    RuntimeCheck(K.unwrap() > 0 && K.unwrap() <= kMaxTopK, "topk must be in (0, 2048]");
    if (B.unwrap() == 0) return;

    const auto params = BF16PagedParams{
        .scores = static_cast<const uint16_t*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = page_table.stride(0),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
    };
    const auto launch = LaunchKernel(static_cast<uint32_t>(B.unwrap()), kBlockSize, device.unwrap());
    if (L.unwrap() <= BF16RegisterShort::kMaxSeqLen) {
      launch.launch(bf16_paged_topk_kernel<0>, params);
    } else if (L.unwrap() <= BF16RegisterLong::kMaxSeqLen) {
      launch.launch(bf16_paged_topk_kernel<1>, params);
    } else {
      launch.launch(bf16_paged_topk_kernel<2>, params);
    }
  }
};

}  // namespace sglang

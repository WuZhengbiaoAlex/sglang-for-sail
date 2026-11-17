#pragma once

#include "topk_v2_bf16.cuh"
namespace sglang {
namespace bf16_split {
namespace tk=device::topk;
struct ValueProblem : tk::TopKProblem {
  uint16_t* values;
  SGL_DEVICE void emit(uint32_t pos,uint32_t idx,uint32_t key) const {
    TopKProblem::emit(pos,idx);
    values[pos]=static_cast<uint16_t>((key&0x8000u)?key^0x8000u:key^0xffffu);
  }
};
struct IndexProblem : tk::TopKProblem {
  const int32_t* indirect;
  SGL_DEVICE void emit(uint32_t pos,uint32_t idx,uint32_t) const {out[pos]=indirect[idx];}
};
struct Params {
  const uint16_t* scores;
  const int32_t* lengths;
  int32_t* indices;
  uint16_t* values;
  int32_t* candidate_lengths;
  const int32_t* indirect;
  int64_t score_stride;
  uint32_t columns,parts,chunk;
};

template<bool Optimized, bool Partial>
__global__ __launch_bounds__(1024,2) void select(const __grid_constant__ Params p) {
  device::enable_smem_spilling();
  using Problem=std::conditional_t<Partial,ValueProblem,IndexProblem>;
  using Short=tk::BF16PagedTopKImpl<1,Problem,3,false,Optimized>;
  using Long=tk::BF16PagedTopKImpl<2,Problem,3,false,Optimized>;
  using Wide=tk::BF16PagedTopKImpl<6,Problem,Optimized?4:3,Optimized,Optimized>;
  using Stream=tk::BF16PagedTopKImpl<0,Problem,3,false,Optimized>;
  using Workspace=std::conditional_t<Optimized && !Partial,
      tk::MaxSmem<typename Short::Smem,typename Long::Smem>,
      tk::MaxSmem<typename Short::Smem,typename Long::Smem,typename Wide::Smem,typename Stream::Smem>>;
  __shared__ Workspace smem;
  const uint32_t row=Partial?blockIdx.x/p.parts:blockIdx.x;
  const uint32_t part=Partial?blockIdx.x%p.parts:0;
  const uint32_t total=p.lengths[row];
  // Preserve the original fixed chunks unless the optimized entry is selected.
  const uint32_t effective_parts=Optimized && Partial && p.parts==6 && total<=147456?3:p.parts;
  const uint32_t chunk=Optimized && Partial?
      max(2048u,((total+effective_parts*8-1)/(effective_parts*8))*8):p.chunk;
  const uint32_t start=part*chunk;
  const uint32_t length=total>start?min(total-start,chunk):0;
  const auto* input=p.scores+row*p.score_stride+start;
  auto* output=p.indices+static_cast<int64_t>(blockIdx.x)*2048;
  if constexpr (Partial) {
    if (part==0 && threadIdx.x==0)
      p.candidate_lengths[row]=min(total/chunk,p.parts)*2048+min(total%chunk,2048u);
  }
  if (length<=2048) {
    for (uint32_t i=threadIdx.x;i<2048;i+=1024) {
      if constexpr (Partial) {
        output[i]=i<length?static_cast<int32_t>(start+i):-1;
        p.values[static_cast<int64_t>(blockIdx.x)*2048+i]=i<length?input[i]:0u;
      } else output[i]=i<length?p.indirect[row*p.columns+i]:-1;
    }
    return;
  }
  tk::TopKProblem base{.in=nullptr,.out=output,.topk=2048,.seq_len=length,.bias=static_cast<int32_t>(start)};
  Problem problem;
  if constexpr (Partial) problem=ValueProblem{base,p.values+static_cast<int64_t>(blockIdx.x)*2048};
  else problem=IndexProblem{base,p.indirect+row*p.columns};
  if (length<=Short::kMaxSeqLen) Short::forward(input,problem,&smem);
  else if constexpr (Optimized && !Partial) {
    // The optimized merge is restricted to <=16384 candidates by its host entry.
    Long::forward(input,problem,&smem);
  } else if (length<=Long::kMaxSeqLen) Long::forward(input,problem,&smem);
  else if constexpr (Optimized) {
    if (length<=Wide::kMaxSeqLen) Wide::forward(input,problem,&smem);
    else Stream::forward(input,problem,&smem);
  } else Stream::forward(input,problem,&smem);
}
// Medium contexts fit in one CTA's private keys; no partial candidates or
// merge launch are needed. Existing long-context specializations stay intact.
__global__ __launch_bounds__(1024, 2) void direct(const __grid_constant__ Params p) {
  device::enable_smem_spilling();
  using Short = tk::BF16PagedTopKImpl<1, tk::TopKProblem, 3, false, true, true>;
  using Long = tk::BF16PagedTopKImpl<2, tk::TopKProblem, 3, false, true, true>;
  using Streaming = tk::BF16PagedTopKImpl<0>;
  using Medium = tk::BF16PagedTopKImpl<4, tk::TopKProblem, 4, true, true, true>;
  __shared__ tk::MaxSmem<Short::Smem, Long::Smem, Streaming::Smem, Medium::Smem> smem;
  const uint32_t row = blockIdx.x, length = p.lengths[row];
  auto* output = p.indices + static_cast<int64_t>(row) * 2048;
  if (length <= 2048) {
    for (uint32_t i = threadIdx.x; i < 2048; i += 1024)
      output[i] = i < length ? static_cast<int32_t>(i) : -1;
    return;
  }
  tk::TopKProblem problem{.in = nullptr, .out = output, .topk = 2048, .seq_len = length};
  const auto* input = p.scores + row * p.score_stride;
  if (length <= Short::kMaxSeqLen) Short::forward(input, problem, &smem);
  else if (length <= Long::kMaxSeqLen) Long::forward(input, problem, &smem);
  // The native streaming selector wins for shorter ordered rows. Choosing
  // inside this kernel follows live lengths, including graph replay.
  else if (length < 30720) Streaming::forward(input, problem, &smem);
  else Medium::forward(input, problem, &smem);
}
}  // namespace bf16_split

template<bool Optimized>
struct PPUTopKSplitBF16Impl {
  static void direct(tvm::ffi::TensorView scores, tvm::ffi::TensorView lengths,
                     tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    auto rows = SymbolicSize{"rows"}, cols = SymbolicSize{"cols"}, stride = SymbolicSize{"stride"};
    TensorMatcher({rows, cols}).with_strides({stride, 1}).with_dtype<bf16_t>().with_device(device).verify(scores);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows, 2048}).with_dtype<int32_t>().with_device(device).verify(output);
    RuntimeCheck(cols.unwrap() <= 32768 && stride.unwrap() % 8 == 0,
                 "direct BF16 TopK requires <=32768 aligned scores");
    const bf16_split::Params p{
        .scores = static_cast<const uint16_t*>(scores.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .indices = static_cast<int32_t*>(output.data_ptr()),
        .values = nullptr, .candidate_lengths = nullptr, .indirect = nullptr,
        .score_stride = scores.stride(0), .columns = static_cast<uint32_t>(cols.unwrap()),
        .parts = 1, .chunk = static_cast<uint32_t>(cols.unwrap()),
    };
    LaunchKernel(rows.unwrap(), 1024, device.unwrap()).launch(bf16_split::direct, p);
  }
  static void partial(tvm::ffi::TensorView scores,tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView indices,tvm::ffi::TensorView values,tvm::ffi::TensorView candidate_lengths,uint32_t parts) {
    using namespace host;
    auto device=SymbolicDevice{};device.set_options<kDLGPU>();
    auto rows=SymbolicSize{"rows"};auto cols=SymbolicSize{"cols"};auto stride=SymbolicSize{"stride"};
    TensorMatcher({rows,cols}).with_strides({stride,1}).with_dtype<bf16_t>().with_device(device).verify(scores);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows,parts*2048}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({rows,parts*2048}).with_dtype<bf16_t>().with_device(device).verify(values);
    const uint32_t chunk=((cols.unwrap()+parts*8-1)/(parts*8))*8;
    RuntimeCheck(chunk>=2048 && chunk<=(Optimized?131968:16384),"invalid chunk");
    const bf16_split::Params p{
      .scores=static_cast<const uint16_t*>(scores.data_ptr()),.lengths=static_cast<const int32_t*>(lengths.data_ptr()),
      .indices=static_cast<int32_t*>(indices.data_ptr()),.values=static_cast<uint16_t*>(values.data_ptr()),
      .candidate_lengths=static_cast<int32_t*>(candidate_lengths.data_ptr()),.indirect=nullptr,
      .score_stride=scores.stride(0),.columns=static_cast<uint32_t>(cols.unwrap()),.parts=parts,.chunk=chunk,
    };
    LaunchKernel(rows.unwrap()*parts,1024,device.unwrap()).launch(bf16_split::select<Optimized,true>,p);
  }
  static void merge(tvm::ffi::TensorView values,tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView indices,tvm::ffi::TensorView output) {
    using namespace host;
    auto device=SymbolicDevice{};device.set_options<kDLGPU>();
    auto rows=SymbolicSize{"rows"};auto cols=SymbolicSize{"cols"};auto stride=SymbolicSize{"stride"};
    TensorMatcher({rows,cols}).with_strides({stride,1}).with_dtype<bf16_t>().with_device(device).verify(values);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows,cols}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({rows,2048}).with_dtype<int32_t>().with_device(device).verify(output);
    if constexpr (Optimized)
      RuntimeCheck(cols.unwrap()<=16384 && stride.unwrap()%8==0,"optimized BF16 merge supports <=16384 aligned candidates");
    const bf16_split::Params p{
      .scores=static_cast<const uint16_t*>(values.data_ptr()),.lengths=static_cast<const int32_t*>(lengths.data_ptr()),
      .indices=static_cast<int32_t*>(output.data_ptr()),.values=nullptr,.candidate_lengths=nullptr,
      .indirect=static_cast<const int32_t*>(indices.data_ptr()),.score_stride=values.stride(0),
      .columns=static_cast<uint32_t>(cols.unwrap()),.parts=1,.chunk=static_cast<uint32_t>(cols.unwrap()),
    };
    LaunchKernel(rows.unwrap(),1024,device.unwrap()).launch(bf16_split::select<Optimized,false>,p);
  }
};
// Preserve the legacy entry; the Python dispatcher selects the optimized alias
// when its shape guards and the opt-in BF16 split switch are satisfied.
using PPUTopKSplitBF16=PPUTopKSplitBF16Impl<false>;
using PPUTopKSplitBF16Optimized=PPUTopKSplitBF16Impl<true>;
}  // namespace sglang

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
// Host-issued cudaMemcpyAsync bindings for IncrementalEPLB weight pulls.

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <pybind11/stl.h>

namespace sglang {
namespace incremental_eplb {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  TORCH_CHECK(result == cudaSuccess, operation, " failed: ",
              cudaGetErrorString(result), " (cudaError ",
              static_cast<int>(result), ")");
}

void check_batch_sizes(const std::vector<int64_t>& src_ptrs,
                       const std::vector<int64_t>& dst_ptrs,
                       const std::vector<int64_t>& sizes) {
  TORCH_CHECK(src_ptrs.size() == dst_ptrs.size() &&
                  src_ptrs.size() == sizes.size(),
              "src_ptrs, dst_ptrs, and sizes must have equal lengths");
}

void check_copy(int64_t src, int64_t dst, int64_t size, size_t index) {
  TORCH_CHECK(src != 0 && dst != 0, "memcpy ", index,
              " has a null pointer");
  TORCH_CHECK(src != dst, "memcpy ", index,
              " has identical source and destination pointers");
  TORCH_CHECK(size > 0, "memcpy ", index,
              " has non-positive size ", size);
}

cudaStream_t resolve_stream(int64_t stream_handle) {
  TORCH_CHECK(stream_handle != 0,
              "IncrementalEPLB copies require a dedicated CUDA stream");
  return reinterpret_cast<cudaStream_t>(
      static_cast<uintptr_t>(stream_handle));
}

}  // namespace

// Enable direct access from the current device to visible peers.
int64_t enable_peer_access() {
  int ndev = 0;
  check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
  int cur = 0;
  check_cuda(cudaGetDevice(&cur), "cudaGetDevice");
  int64_t enabled = 0;
  for (int d = 0; d < ndev; ++d) {
    if (d == cur) continue;
    int can = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can, cur, d),
               "cudaDeviceCanAccessPeer");
    if (!can) {
      TORCH_WARN("direct peer access is unavailable for device ", cur, "->", d,
                 "; cudaMemcpyAsync may use a slower staged route");
      continue;
    }
    cudaError_t e = cudaDeviceEnablePeerAccess(d, 0);
    if (e == cudaSuccess) ++enabled;
    else if (e != cudaErrorPeerAccessAlreadyEnabled)
      TORCH_WARN("peer access ", cur, "->", d, " failed: ", cudaGetErrorString(e));
    cudaGetLastError();
  }
  return enabled;
}

// Copies from different source devices use independent streams.
std::vector<cudaStream_t>& peer_streams() {
  static std::vector<cudaStream_t> st;
  if (st.empty()) {
    int ndev = 0;
    check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    st.resize(ndev > 0 ? ndev : 1);
    for (auto& x : st)
      check_cuda(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags(peer)");
  }
  return st;
}

cudaEvent_t kickoff_event() {
  static cudaEvent_t event = nullptr;
  if (event == nullptr) {
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
               "cudaEventCreateWithFlags(peer kickoff)");
  }
  return event;
}

std::vector<cudaEvent_t>& join_events() {
  static std::vector<cudaEvent_t> ev;
  const size_t count = peer_streams().size();
  if (ev.empty()) {
    ev.resize(count);
    for (auto& e : ev)
      check_cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming),
                 "cudaEventCreateWithFlags(peer join)");
  }
  return ev;
}

void prepare_copy_engine() {
  (void)peer_streams();
  (void)kickoff_event();
  (void)join_events();
}

// Cache source-device lookup for the process-lifetime arena addresses.
int device_of(int64_t ptr) {
  static std::unordered_map<int64_t, int> cache;
  auto it = cache.find(ptr);
  if (it != cache.end()) return it->second;
  cudaPointerAttributes at{};
  int dev = -1;
  if (cudaPointerGetAttributes(&at, reinterpret_cast<void*>(
          static_cast<uintptr_t>(ptr))) == cudaSuccess)
    dev = at.device;
  cudaGetLastError();
  cache[ptr] = dev;
  return dev;
}

void enqueue_memcpy_batch(const std::vector<int64_t>& src_ptrs,
                          const std::vector<int64_t>& dst_ptrs,
                          const std::vector<int64_t>& sizes,
                          int64_t stream_handle) {
  check_batch_sizes(src_ptrs, dst_ptrs, sizes);
  cudaStream_t caller = resolve_stream(stream_handle);
  auto& st = peer_streams();
  std::vector<char> used(st.size(), 0);

  // Validate the whole batch before enqueuing any part of it. A bad trailing
  // entry must not leave a partially written expert generation behind.
  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    check_copy(src_ptrs[i], dst_ptrs[i], sizes[i], i);
  }

  int current_device = 0;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");

  // Propagate copy-stream ordering to the fan-out streams. Prior model reads
  // are fenced by the planner publication before this host call.
  cudaEvent_t kickoff = kickoff_event();
  check_cuda(cudaEventRecord(kickoff, caller), "cudaEventRecord(peer kickoff)");

  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    int d = device_of(src_ptrs[i]);
    cudaStream_t s =
        (d >= 0 && d != current_device && (size_t)d < st.size()) ? st[d]
                                                                  : caller;
    if (s != caller && !used[d]) {
      check_cuda(cudaStreamWaitEvent(s, kickoff, 0),
                 "cudaStreamWaitEvent(peer kickoff)");
      used[d] = 1;
    }
    check_cuda(
        cudaMemcpyAsync(
            reinterpret_cast<void*>(static_cast<uintptr_t>(dst_ptrs[i])),
            reinterpret_cast<const void*>(
                static_cast<uintptr_t>(src_ptrs[i])),
            static_cast<size_t>(sizes[i]), cudaMemcpyDefault, s),
        "cudaMemcpyAsync");
  }

  // Join fan-out streams so one event on the caller covers the full batch.
  auto& ev = join_events();
  for (size_t d = 0; d < st.size(); ++d) {
    if (!used[d]) continue;
    check_cuda(cudaEventRecord(ev[d], st[d]), "cudaEventRecord(peer)");
    check_cuda(cudaStreamWaitEvent(caller, ev[d], 0), "cudaStreamWaitEvent(peer)");
  }
}

}  // namespace incremental_eplb
}  // namespace sglang

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enable_peer_access", &sglang::incremental_eplb::enable_peer_access,
        "Enable direct P2P from this device to all peers");
  m.def("prepare_copy_engine", &sglang::incremental_eplb::prepare_copy_engine,
        "Create the process-lifetime peer streams and events");
  m.def("enqueue_memcpy_batch",
        &sglang::incremental_eplb::enqueue_memcpy_batch,
        "Enqueue a batch of D2D cudaMemcpyAsync operations",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}

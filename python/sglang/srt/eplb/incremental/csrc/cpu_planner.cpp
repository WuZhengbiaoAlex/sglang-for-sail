// Copyright 2026 SGLang Team. Licensed under the Apache License, Version 2.0.
#include <ATen/MemoryOverlap.h>
#include <omp.h>
#include <torch/extension.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
using std::max;
using std::min;
namespace {
constexpr int MAX_E = 1024, MAX_R = 64, MAX_P = MAX_E + 2 * MAX_R;
constexpr int64_t MAX_TOKENS = 1LL << 24;
struct Workspace {
  uint64_t old_rank_mask[MAX_E], resident_rank_mask[MAX_E];
  int resident[MAX_P], occupancy[MAX_R], copy_source[MAX_P];
  // One linked-list entry per physical slot, not an E x R shared array.
  short head[MAX_E], next[MAX_P];
  unsigned char count[MAX_E];
  int64_t load[MAX_R];
  bool selected[MAX_P], protected_slot[MAX_P], changed[MAX_E];
};

int owner_at(const Workspace &w, int e, int i) {
  int s = w.head[e];
  while (i-- > 0)
    s = w.next[s];
  return s;
}
void append_owner(Workspace &w, int e, int s) {
  w.next[s] = -1;
  if (w.count[e])
    w.next[owner_at(w, e, w.count[e] - 1)] = s;
  else
    w.head[e] = s;
  ++w.count[e];
}
void replace_owner(Workspace &w, int e, int i, int dst) {
  int src = owner_at(w, e, i);
  w.next[dst] = w.next[src];
  if (i)
    w.next[owner_at(w, e, i - 1)] = dst;
  else
    w.head[e] = dst;
}
bool owns_rank(const Workspace &w, int e, int r, int S) {
  for (int s = w.head[e]; s >= 0; s = w.next[s])
    if (s / S == r)
      return true;
  return false;
}
int64_t part(int64_t weight, int count, int i) {
  return weight / count + (i < weight % count);
}
int required_count(int64_t weight, int64_t total, int R, int numerator,
                   int denominator) {
  int64_t demand = weight * R * denominator, capacity = total * numerator;
  return min(static_cast<int64_t>(R),
             max(int64_t(1), (demand + capacity - 1) / capacity));
}
int desired_count(int64_t weight, int64_t total, int R, int current,
                  int numerator, int denominator, int merge_numerator,
                  int merge_denominator) {
  if (!total || !weight)
    return 1;
  int grow = required_count(weight, total, R, numerator, denominator);
  if (grow > current)
    return grow;
  int shrink =
      required_count(weight, total, R, merge_numerator, merge_denominator);
  return shrink < current ? shrink : current;
}

int find_destination(Workspace &w, int e, int r, int S, int copies, int budget,
                     int &cost) {
  if (owns_rank(w, e, r, S))
    return -1;
  if (w.resident_rank_mask[e] & (uint64_t(1) << r))
    for (int s = r * S; s < (r + 1) * S; ++s)
      if (w.resident[s] == e && !w.selected[s]) {
        cost = 0;
        return s;
      }
  if (w.old_rank_mask[e] & (uint64_t(1) << r))
    return -1;
  if (copies >= budget)
    return -1;
  int best = -1;
  for (int s = r * S; s < (r + 1) * S; ++s) {
    if (w.protected_slot[s] || w.selected[s] || w.copy_source[s] >= 0)
      continue;
    if (best < 0 || (w.resident[best] >= 0 && w.resident[s] < 0))
      best = s;
  }
  cost = 1;
  return best;
}

void install(Workspace &w, const int *old_owners, int e, int dst, int S, int R,
             int ranks_per_node, int cost, int &copies) {
  if (!cost)
    return;
  int src = old_owners[R * e];
  int node = dst / S / ranks_per_node;
  for (int i = 1; i < R && old_owners[R * e + i] >= 0; ++i) {
    int candidate = old_owners[R * e + i];
    int a = src / S / ranks_per_node != node;
    int b = candidate / S / ranks_per_node != node;
    if (b < a || (a == b && candidate < src))
      src = candidate;
  }
  w.copy_source[dst] = src;
  w.resident[dst] = e;
  // A conservative presence mask: stale bits only cause an extra exact scan.
  w.resident_rank_mask[e] |= uint64_t(1) << (dst / S);
  ++copies;
}

void plan_split_merge(const int64_t *tokens, const int *old,
                      const int *old_owners, int *output, int *owners_out,
                      int *sources_out, int E, int R, int S, int budget,
                      int rpn, int numerator, int denominator,
                      int merge_numerator, int merge_denominator,
                      int layer_offset) {
  const int P = R * S, l = layer_offset;
  tokens += l * E;
  old += l * P;
  old_owners += l * E * R;
  output += l * P;
  owners_out += l * E * R;
  sources_out += l * P;
  Workspace w;
  if (0 == 0) {
    int copies = 0;
    int64_t total = 0;
    for (int r = 0; r < R; ++r) {
      w.load[r] = 0;
      w.occupancy[r] = 0;
    }
    for (int s = 0; s < P; ++s) {
      w.resident[s] = old[s];
      w.selected[s] = w.protected_slot[s] = false;
      w.copy_source[s] = -1;
    }
    for (int e = 0; e < E; ++e) {
      w.old_rank_mask[e] = w.resident_rank_mask[e] = 0;
    }
    for (int s = 0; s < P; ++s) {
      if (old[s] >= 0)
        w.old_rank_mask[old[s]] |= uint64_t(1) << (s / S);
      if (w.resident[s] >= 0)
        w.resident_rank_mask[w.resident[s]] |= uint64_t(1) << (s / S);
    }
    for (int e = 0; e < E; ++e) {
      assert(tokens[e] >= 0 && tokens[e] <= MAX_TOKENS);

      total += tokens[e];
      w.changed[e] = false;
      w.head[e] = -1;
      w.count[e] = 0;
      for (int i = 0; i < R && old_owners[e * R + i] >= 0; ++i) {
        int s = old_owners[e * R + i];
        assert(s < P && old[s] == e && !w.selected[s]);
        assert(!owns_rank(w, e, s / S, S));
        append_owner(w, e, s);
        w.selected[s] = w.protected_slot[s] = true;
        ++w.occupancy[s / S];
      }
      assert(w.count[e] > 0);
      for (int i = 0; i < w.count[e]; ++i)
        w.load[owner_at(w, e, i) / S] += part(tokens[e], w.count[e], i);
    }
    assert(total <= MAX_TOKENS); // Bounds the integer objective products.
    for (int e = 0; e < E; ++e) {
      int desired =
          desired_count(tokens[e], total, R, w.count[e], numerator, denominator,
                        merge_numerator, merge_denominator);
      if (desired >= w.count[e])
        continue;
      int slots[MAX_R], n = w.count[e];
      for (int i = 0; i < n; ++i) {
        slots[i] = owner_at(w, e, i);
        w.load[slots[i] / S] -= part(tokens[e], n, i);
        w.selected[slots[i]] = false;
        --w.occupancy[slots[i] / S];
      }
      // Keep the least loaded existing owners; retired storage stays pinned
      // until the previous generation's decode and prefill reads finish.
      for (int i = 1; i < n; ++i) {
        int s = slots[i], j = i;
        while (j > 0 && (w.load[slots[j - 1] / S] > w.load[s / S] ||
                         (w.load[slots[j - 1] / S] == w.load[s / S] &&
                          slots[j - 1] > s))) {
          slots[j] = slots[j - 1];
          --j;
        }
        slots[j] = s;
      }
      w.head[e] = -1;
      w.count[e] = 0;
      for (int i = 0; i < desired; ++i) {
        append_owner(w, e, slots[i]);
        w.selected[slots[i]] = true;
        ++w.occupancy[slots[i] / S];
        w.load[slots[i] / S] += part(tokens[e], desired, i);
      }
      w.changed[e] = true;
    }
    bool considered[MAX_E] = {};
    for (int it = 0; it < E; ++it) {
      int e = -1;
      for (int j = 0; j < E; ++j)
        if (!considered[j] &&
            w.count[j] < desired_count(tokens[j], total, R, w.count[j],
                                       numerator, denominator, merge_numerator,
                                       merge_denominator) &&
            (e < 0 || tokens[j] > tokens[e]))
          e = j;
      if (e < 0)
        break;
      considered[e] = true;
      int desired =
          desired_count(tokens[e], total, R, w.count[e], numerator, denominator,
                        merge_numerator, merge_denominator);
      while (w.count[e] < desired) {
        int n = w.count[e], best = -1, best_cost = 2;
        int64_t weight = part(tokens[e], n + 1, n), released = 0;
        for (int i = 0; i < n; ++i) {
          int r = owner_at(w, e, i) / S;
          int64_t d = part(tokens[e], n, i) - part(tokens[e], n + 1, i);
          released += 2 * w.load[r] * d - d * d;
        }
        int64_t best_gain = LLONG_MIN;
        for (int r = 0; r < R; ++r) {
          if (w.occupancy[r] >= S - 1)
            continue;
          int cost = 0,
              dst = find_destination(w, e, r, S, copies, budget, cost);
          if (dst < 0)
            continue;
          int64_t gain = released - 2 * w.load[r] * weight - weight * weight;
          if (best < 0 || gain > best_gain ||
              (gain == best_gain && cost < best_cost)) {
            best = dst;
            best_gain = gain;
            best_cost = cost;
          }
        }
        if (best < 0)
          break;
        install(w, old_owners, e, best, S, R, rpn, best_cost, copies);
        for (int i = 0; i < n; ++i)
          w.load[owner_at(w, e, i) / S] -=
              part(tokens[e], n, i) - part(tokens[e], n + 1, i);
        append_owner(w, e, best);
        w.selected[best] = true;
        ++w.occupancy[best / S];
        w.load[best / S] += weight;
        w.changed[e] = true;
      }
    }
  }

  // Write the wide owner table cooperatively; planning decisions stay serial.
  for (int s = 0; s < P; s++) {
    output[s] = w.resident[s];
    sources_out[s] = w.copy_source[s];
  }
  for (int e = 0; e < E; ++e)
    for (int i = 0; i < R; i++)
      owners_out[R * e + i] = i < w.count[e] ? owner_at(w, e, i) : -1;
}

struct Exchange {
  int e, i, dst, f, j, back, ce, cf, maximum;
  int64_t gain;
};

Exchange empty_exchange() {
  Exchange result = {};
  result.e = result.f = -1;
  result.dst = result.back = -1;
  result.maximum = INT_MAX;
  return result;
}

bool better_exchange(const Exchange &a, const Exchange &b) {
  if (a.e < 0)
    return false;
  if (b.e < 0)
    return true;
  if (a.maximum != b.maximum)
    return a.maximum < b.maximum;
  if (a.gain != b.gain)
    return a.gain > b.gain;
  if (a.ce + a.cf != b.ce + b.cf)
    return a.ce + a.cf < b.ce + b.cf;
  // Source slots are uniquely determined by (expert, owner index).
  const int ka[] = {a.e, a.i, a.dst, a.f, a.j, a.back};
  const int kb[] = {b.e, b.i, b.dst, b.f, b.j, b.back};
  for (int k = 0; k < 6; ++k)
    if (ka[k] != kb[k])
      return ka[k] < kb[k];
  return false;
}

int64_t excess(int64_t load, int64_t lower) {
  int64_t d = load > lower ? load - lower : 0;
  return d * d;
}

void repair_exchange(const int64_t *tokens, const int *old,
                     const int *old_owners, int *output, int *owners_out,
                     int *sources_out, int E, int R, int S, int budget,
                     int move_budget, int rpn, int gain_divisor,
                     int layer_offset) {
  const int l = layer_offset, P = R * S;
  tokens += l * E;
  old += l * P;
  old_owners += l * E * R;
  output += l * P;
  owners_out += l * E * R;
  sources_out += l * P;
  Workspace w;
  int slot_part[MAX_P], slot_weight[MAX_P], copies, moves, source_order[MAX_R];
  int64_t total, lower, minimum_gain;
  Exchange selected;

  copies = 0;
  moves = 0;
  total = 0;
  lower = 0;
  for (int r = 0; r < R; ++r) {
    w.load[r] = 0;
    w.occupancy[r] = 0;
  }
  for (int s = 0; s < P; ++s) {
    w.resident[s] = output[s];
    w.copy_source[s] = sources_out[s];
    copies += sources_out[s] >= 0;
    w.selected[s] = false;
    w.protected_slot[s] = false;
    slot_part[s] = -1;
    slot_weight[s] = 0;
  }
  for (int e = 0; e < E; ++e) {
    w.old_rank_mask[e] = w.resident_rank_mask[e] = 0;
  }
  for (int s = 0; s < P; ++s) {
    if (old[s] >= 0)
      w.old_rank_mask[old[s]] |= uint64_t(1) << (s / S);
    if (w.resident[s] >= 0)
      w.resident_rank_mask[w.resident[s]] |= uint64_t(1) << (s / S);
  }
  for (int e = 0; e < E; ++e) {
    total += tokens[e];
    w.head[e] = -1;
    w.count[e] = 0;
    w.changed[e] = false;
    for (int i = 0; i < R; ++i) {
      int s = owners_out[R * e + i], prior = old_owners[R * e + i];
      w.changed[e] |= s != prior;
      // Both owner lists are contiguous and padded with negative entries.
      if (s < 0 && prior < 0)
        break;
      if (prior >= 0)
        w.protected_slot[prior] = true;
      if (s < 0)
        continue;
      append_owner(w, e, s);
      w.selected[s] = true;
      slot_part[s] = MAX_R * e + i;
      ++w.occupancy[s / S];
    }
    int n = w.count[e];
    int64_t bound = (tokens[e] + n - 1) / n;
    if (bound > lower)
      lower = bound;
    for (int i = 0; i < n; ++i) {
      int s = owner_at(w, e, i);
      slot_weight[s] = part(tokens[e], n, i);
      w.load[s / S] += slot_weight[s];
    }
  }
  int64_t average = (total + R - 1) / R;
  if (average > lower)
    lower = average;
  // Skip negligible improvements that would consume the bounded copy budget.
  int64_t denominator = int64_t(gain_divisor) * R * R;
  minimum_gain = (total * total + denominator - 1) / denominator;

  for (int iteration = 0; iteration < move_budget; ++iteration) {
    for (int r = 0; r < R; ++r)
      source_order[r] = r;
    for (int i = 1; i < R; ++i) {
      int r = source_order[i], j = i;
      while (j > 0 && (w.load[source_order[j - 1]] < w.load[r] ||
                       (w.load[source_order[j - 1]] == w.load[r] &&
                        source_order[j - 1] > r))) {
        source_order[j] = source_order[j - 1];
        --j;
      }
      source_order[j] = r;
    }
    selected = empty_exchange();

    // Repair the most overloaded feasible source first. Fall back only when
    // residency, copy budget, or pinned slots prevent any improving exchange.
    for (int source_attempt = 0; source_attempt < R; ++source_attempt) {
      int src = source_order[source_attempt];
      if (w.load[src] <= lower)
        break;
      Exchange best = empty_exchange();
      for (int job = 0; job < S * R; job++) {
        int s = src * S + job / R, dst = job % R;
        if (w.load[src] <= w.load[dst])
          continue;
        int64_t old_excess =
            excess(w.load[src], lower) + excess(w.load[dst], lower);
        if (old_excess < minimum_gain)
          continue;
        int64_t unaffected = 0;
        for (int h = 0; h < 3 && h < R; ++h) {
          int r = source_order[h];
          if (r != src && r != dst) {
            unaffected = w.load[r];
            break;
          }
        }
        if (best.e >= 0 && unaffected > best.maximum)
          continue;
        int p = slot_part[s];
        if (p < 0 || w.changed[p / MAX_R])
          continue;
        int e = p / MAX_R, i = p % MAX_R;
        int64_t weight = slot_weight[s];
        if (!weight)
          continue;
        int ce = 0;
        int to = find_destination(w, e, dst, S, copies, budget, ce);
        if (to < 0)
          continue;
        for (int v = -1; v < S; ++v) {
          int q = v < 0 ? -1 : slot_part[dst * S + v];
          if (v < 0 ? w.occupancy[dst] >= S - 1
                    : (q < 0 || w.changed[q / MAX_R]))
            continue;
          int f = q < 0 ? -1 : q / MAX_R, j = q < 0 ? 0 : q % MAX_R;
          if (f == e || moves + (f < 0 ? 1 : 2) > move_budget)
            continue;
          int64_t other = f < 0 ? 0 : slot_weight[dst * S + v],
                  delta = weight - other;
          if (delta <= 0)
            continue;
          int64_t variance_gain =
              2 * delta * (w.load[src] - w.load[dst] - delta);
          if (variance_gain <= 0 || variance_gain < minimum_gain)
            continue;
          int64_t gain = old_excess - excess(w.load[src] - delta, lower) -
                         excess(w.load[dst] + delta, lower);
          if (gain <= 0 || gain < minimum_gain)
            continue;
          int64_t maximum = unaffected;
          if (w.load[src] - delta > maximum)
            maximum = w.load[src] - delta;
          if (w.load[dst] + delta > maximum)
            maximum = w.load[dst] + delta;
          if (best.e >= 0 && (maximum > best.maximum ||
                              (maximum == best.maximum && gain < best.gain)))
            continue;
          int cf = 0;
          int back =
              f < 0 ? -1 : find_destination(w, f, src, S, copies, budget, cf);
          if ((f >= 0 && back < 0) || copies + ce + cf > budget)
            continue;
          Exchange candidate = empty_exchange();
          candidate.e = e;
          candidate.i = i;
          candidate.dst = to;
          candidate.f = f;
          candidate.j = j;
          candidate.back = back;
          candidate.ce = ce;
          candidate.cf = cf;
          candidate.maximum = maximum;
          candidate.gain = gain;
          if (better_exchange(candidate, best))
            best = candidate;
        }
      }
      selected = best;

      bool found = selected.e >= 0;

      if (found)
        break;
    }

    Exchange best = selected;
    if (best.e >= 0) {
      int source = owner_at(w, best.e, best.i);
      int other = best.f < 0 ? -1 : owner_at(w, best.f, best.j);
      int source_weight = slot_weight[source];
      int other_weight = other < 0 ? 0 : slot_weight[other];
      int64_t delta = source_weight - other_weight;
      install(w, old_owners, best.e, best.dst, S, R, rpn, best.ce, copies);
      replace_owner(w, best.e, best.i, best.dst);
      w.changed[best.e] = true;
      ++moves;
      w.selected[source] = false;
      w.selected[best.dst] = true;
      slot_part[source] = -1;
      slot_part[best.dst] = MAX_R * best.e + best.i;
      slot_weight[source] = 0;
      slot_weight[best.dst] = source_weight;
      --w.occupancy[source / S];
      ++w.occupancy[best.dst / S];
      if (best.f >= 0) {
        install(w, old_owners, best.f, best.back, S, R, rpn, best.cf, copies);
        replace_owner(w, best.f, best.j, best.back);
        w.changed[best.f] = true;
        ++moves;
        w.selected[other] = false;
        w.selected[best.back] = true;
        slot_part[other] = -1;
        slot_part[best.back] = MAX_R * best.f + best.j;
        slot_weight[other] = 0;
        slot_weight[best.back] = other_weight;
        --w.occupancy[other / S];
        ++w.occupancy[best.back / S];
      }
      w.load[source / S] -= delta;
      w.load[best.dst / S] += delta;
    }

    bool done = selected.e < 0;

    if (done)
      break;
  }
  for (int s = 0; s < P; s++) {
    output[s] = w.resident[s];
    sources_out[s] = w.copy_source[s];
  }
  for (int e = 0; e < E; e++)
    for (int i = 0; i < R; ++i)
      owners_out[R * e + i] = i < w.count[e] ? owner_at(w, e, i) : -1;
}
} // namespace

bool plan_into(torch::Tensor tokens, torch::Tensor resident,
               torch::Tensor owners, torch::Tensor output,
               torch::Tensor owners_out, torch::Tensor sources_out,
               int64_t ranks, int64_t copy_budget, int64_t move_budget,
               int64_t ranks_per_node, int64_t split_numerator,
               int64_t split_denominator, int64_t merge_numerator,
               int64_t merge_denominator, int64_t gain_divisor,
               int64_t threads) {
  for (const auto &t :
       {tokens, resident, owners, output, owners_out, sources_out})
    TORCH_CHECK(t.is_cpu() && t.is_contiguous(),
                "Contiguous CPU tensors required");
  TORCH_CHECK(tokens.scalar_type() == torch::kInt64, "Expected int64 counts");
  for (const auto &t : {resident, owners, output, owners_out, sources_out})
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "Expected int32 layouts");
  TORCH_CHECK(tokens.dim() == 2 && resident.dim() == 2 && owners.dim() == 3,
              "Invalid dimensions");
  const int64_t L = tokens.size(0), E = tokens.size(1), P = resident.size(1);
  TORCH_CHECK(L > 0 && E > 0 && E <= MAX_E && ranks > 0 && ranks <= MAX_R &&
                  P > 0 && P <= MAX_P && P % ranks == 0 && P / ranks <= 64 &&
                  L <= INT_MAX / (E * ranks) && L <= INT_MAX / P,
              "Unsupported planner geometry");
  TORCH_CHECK(resident.size(0) == L && owners.size(0) == L &&
                  owners.size(1) == E && owners.size(2) == ranks &&
                  output.sizes() == resident.sizes() &&
                  sources_out.sizes() == resident.sizes() &&
                  owners_out.sizes() == owners.sizes(),
              "Shape mismatch");
  TORCH_CHECK(split_numerator > 0 && split_numerator <= split_denominator &&
                  split_denominator <= 1000000,
              "Split target must be in (0, 1]");
  TORCH_CHECK(merge_numerator > 0 && merge_numerator < merge_denominator &&
                  merge_denominator <= 1000000 &&
                  merge_numerator * split_denominator <
                      split_numerator * merge_denominator,
              "Merge target must be positive and below split target");
  TORCH_CHECK(copy_budget >= 0 && copy_budget <= P && move_budget >= 0 &&
                  move_budget <= MAX_E && ranks_per_node > 0 &&
                  ranks % ranks_per_node == 0 && gain_divisor > 0 &&
                  gain_divisor <= INT_MAX && threads > 0 && threads <= INT_MAX,
              "Invalid planner bounds");
  for (const auto &out : {output, owners_out, sources_out})
    for (const auto &in : {tokens, resident, owners})
      at::assert_no_overlap(out, in);
  at::assert_no_overlap(output, owners_out);
  at::assert_no_overlap(output, sources_out);
  at::assert_no_overlap(owners_out, sources_out);

  const auto *counts = tokens.data_ptr<int64_t>();
  const auto *physical = resident.data_ptr<int>();
  const auto *active = owners.data_ptr<int>();
  const int S = P / ranks;
  // Reject malformed snapshots before entering OpenMP: exceptions must not
  // escape a parallel region, and owner indices are used as native addresses.
  for (int l = 0; l < L; ++l) {
    int64_t total = 0;
    bool selected[MAX_P] = {};
    int occupancy[MAX_R] = {};
    for (int s = 0; s < P; ++s)
      TORCH_CHECK(physical[l * P + s] >= -1 && physical[l * P + s] < E,
                  "Resident expert outside logical range");
    for (int e = 0; e < E; ++e) {
      const auto n = counts[l * E + e];
      TORCH_CHECK(n >= 0 && n <= MAX_TOKENS,
                  "Token count outside supported range");
      total += n;
      uint64_t used_ranks = 0;
      bool padding = false;
      for (int i = 0; i < ranks; ++i) {
        const int slot = active[(l * E + e) * ranks + i];
        if (slot == -1) {
          padding = true;
          continue;
        }
        TORCH_CHECK(!padding && slot >= 0 && slot < P,
                    "Invalid owner slot or padding");
        TORCH_CHECK(physical[l * P + slot] == e && !selected[slot],
                    "Owner does not match resident storage");
        const int rank = slot / S;
        const auto bit = uint64_t(1) << rank;
        TORCH_CHECK(!(used_ranks & bit), "Duplicate owner rank");
        used_ranks |= bit;
        selected[slot] = true;
        ++occupancy[rank];
      }
      TORCH_CHECK(used_ranks != 0, "Every expert needs an active owner");
    }
    TORCH_CHECK(total <= MAX_TOKENS,
                "Total token count exceeds supported range");
    for (int r = 0; r < ranks; ++r)
      TORCH_CHECK(occupancy[r] < S, "Each rank needs an inactive staging slot");
  }
  const int workers = std::min(threads, L);
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int l = 0; l < L; ++l) {
    plan_split_merge(counts, physical, active, output.data_ptr<int>(),
                     owners_out.data_ptr<int>(), sources_out.data_ptr<int>(), E,
                     ranks, S, copy_budget, ranks_per_node, split_numerator,
                     split_denominator, merge_numerator, merge_denominator, l);
    repair_exchange(counts, physical, active, output.data_ptr<int>(),
                    owners_out.data_ptr<int>(), sources_out.data_ptr<int>(), E,
                    ranks, S, copy_budget, move_budget, ranks_per_node,
                    gain_divisor, l);
  }
  // A resident copy can require work even without a routing-table change.
  const auto *copies = sources_out.data_ptr<int>();
  const auto *next = owners_out.data_ptr<int>();
  for (int64_t i = 0; i < L * P; ++i)
    if (copies[i] >= 0)
      return true;
  for (int64_t i = 0; i < L * E * ranks; ++i)
    if (active[i] != next[i])
      return true;
  return false;
}

void prepare_pulls(torch::Tensor sources, torch::Tensor old_owners,
                   torch::Tensor resident, torch::Tensor next_resident,
                   torch::Tensor pulls, torch::Tensor experts, int64_t ranks,
                   int64_t rank) {
  for (const auto &t :
       {sources, old_owners, resident, next_resident, pulls, experts})
    TORCH_CHECK(t.is_cpu() && t.is_contiguous() &&
                    t.scalar_type() == torch::kInt32,
                "Contiguous CPU int32 tensors required");
  TORCH_CHECK(sources.dim() == 2 && old_owners.dim() == 3,
              "Invalid publication dimensions");
  const int64_t L = sources.size(0), P = sources.size(1);
  const int64_t E = old_owners.size(1);
  TORCH_CHECK(L > 0 && E > 0 && E <= MAX_E && P > 0 && P <= MAX_P &&
                  ranks > 0 && ranks <= MAX_R && P % ranks == 0 && rank >= 0 &&
                  rank < ranks && P / ranks <= 64 &&
                  L <= INT_MAX / (E * ranks) && L <= INT_MAX / P,
              "Invalid publication geometry");
  const int S = P / ranks;
  TORCH_CHECK(
      old_owners.size(0) == L && old_owners.size(2) == ranks &&
          resident.sizes() == sources.sizes() &&
          next_resident.sizes() == sources.sizes() && pulls.dim() == 3 &&
          pulls.size(0) == L && pulls.size(1) == S && pulls.size(2) == 4 &&
          experts.dim() == 2 && experts.size(0) == L && experts.size(1) == S,
      "Publication shape mismatch");
  for (const auto &out : {pulls, experts})
    for (const auto &in : {sources, old_owners, resident, next_resident})
      at::assert_no_overlap(out, in);
  at::assert_no_overlap(pulls, experts);
  const int *srcs = sources.data_ptr<int>(), *old = old_owners.data_ptr<int>();
  const int *res = resident.data_ptr<int>();
  const int *next_res = next_resident.data_ptr<int>();
  int *records = pulls.data_ptr<int>(), *ids = experts.data_ptr<int>();
  // Validate every rank's copies before exposing any local task records.
  for (int l = 0; l < L; ++l) {
    for (int dst = 0; dst < P; ++dst) {
      const int i = l * P + dst, src = srcs[i];
      TORCH_CHECK(src >= -1, "Invalid copy source sentinel");
      if (src >= 0) {
        TORCH_CHECK(src < P && src != dst, "Invalid copy source");
        const int e = next_res[i];
        TORCH_CHECK(e >= 0 && e < E && res[l * P + src] == e,
                    "Copy source expert mismatch");
        bool found = false;
        for (int j = 0; j < ranks; ++j)
          found |= old[(l * E + e) * ranks + j] == src;
        TORCH_CHECK(found, "Copy source is not an active owner");
        const int previous = res[i];
        TORCH_CHECK(previous >= -1 && previous < E,
                    "Invalid destination resident");
        if (previous >= 0)
          for (int j = 0; j < ranks; ++j)
            TORCH_CHECK(old[(l * E + previous) * ranks + j] != dst,
                        "Copy destination is still active");
      }
    }
  }
  for (int l = 0; l < L; ++l) {
    for (int local = 0; local < S; ++local) {
      const int i = l * P + rank * S + local, src = srcs[i];
      const int n = l * S + local;
      records[4 * n] = src < 0 ? -1 : src / S;
      records[4 * n + 1] = src < 0 ? -1 : src % S;
      records[4 * n + 2] = local;
      records[4 * n + 3] = l;
      ids[n] = src < 0 ? -1 : next_res[i];
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_pulls", &prepare_pulls,
        "Validate and prepare CPU expert pulls",
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("plan_into", &plan_into, "Plan placement into reusable CPU buffers",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}

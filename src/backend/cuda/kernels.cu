// hessboost CUDA kernels: device-resident histogram tree growth whose
// histograms reproduce the CPU backend's `f64` sums bit for bit
// (`src/backend/cuda/mod.rs` has the design and the exactness argument).
//
// Compiled at run time by NVRTC for the device's own architecture, with FP
// contraction off (`--fmad=false`), no flush-to-zero, and IEEE division and
// square root, so every floating-point operation here is the single IEEE
// operation the CPU performs. No libdevice transcendentals and no
// floating-point atomics: integer sums are order-free, and every `f64` sum
// is one thread's chain in the CPU's order.
//
// Layouts (row ids `u32`, element offsets 64-bit):
// - bins: row-major ELLPACK, `n_cols` feature-local bins per row (`u8`,
//   `u16` or `u32`); `sentinel` marks a missing value (a dense index passes
//   a sentinel no stored value equals).
// - `feature_first`: each feature's first global bin.
// - `gpair`: one `float2` (gradient, Hessian) per row, as `GradPair`.
// - `units`: one `longlong2` per row, the pair in integer grains.
// - integer histograms: `[slot][bin][2]` 64-bit words, two's complement.
// - `f64` histograms: `[slot][bin]` `double2`, as `GradStats`.

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define FULL_MASK 0xffffffffu

#define GRID_STRIDE(i, n)                                                  \
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (n);      \
         i += (u64)gridDim.x * blockDim.x)

// ---------------------------------------------------------------------------
// Gradients and rows

// Each gradient pair as integer multiples of its component's grain:
// `(f64)x * 2^-grain`, an exact power-of-two scaling, truncated to `i64`
// (exact for every slice the exact paths read).
extern "C" __global__ void stage_units(const float2* __restrict__ gpair,
                                       longlong2* __restrict__ units, u64 n,
                                       double grad_scale, double hess_scale) {
    GRID_STRIDE(i, n) {
        float2 p = gpair[i];
        units[i] = make_longlong2((i64)((double)p.x * grad_scale),
                                  (i64)((double)p.y * hess_scale));
    }
}

// `rows[i] = first + i`: an unsampled tree's rows without an upload.
extern "C" __global__ void iota_rows(u32* __restrict__ rows, u64 n, u32 first) {
    GRID_STRIDE(i, n) { rows[i] = first + (u32)i; }
}

// `reg:squarederror`'s gradients from the device margins, the CPU's
// operations in `f32`: `w` (the row weight, or 1), times
// `scale_pos_weight` for a label of exactly 1, then `((p - y) * w, w)`.
extern "C" __global__ void squared_error(const float* __restrict__ margins,
                                         const float* __restrict__ labels,
                                         const float* __restrict__ weights,
                                         int weighted, float scale_pos_weight,
                                         u64 n, float2* __restrict__ gpair) {
    GRID_STRIDE(i, n) {
        float p = margins[i], y = labels[i];
        float w = weighted ? weights[i] : 1.0f;
        if (y == 1.0f) w = w * scale_pos_weight;
        gpair[i] = make_float2((p - y) * w, w);
    }
}

// The host vector kernels' exponential (`simd/x86_64.rs` `exp_f32`,
// `simd/aarch64.rs` `expq_f32::<true>`, identical operations) for
// `|v| <= 80`: range reduction by a split ln 2, Estrin's seventh-order
// polynomial, the `2^e` scaling. Explicit `fmaf` where the host fuses;
// everything else is a separate IEEE operation (`--fmad=false`).
__device__ __forceinline__ float exp_vector(float v) {
    float scaled = v * __uint_as_float(0x3fb8aa3bu);  // log2(e)
    int e = __float2int_rn(scaled);
    float ef = (float)e;
    float r = fmaf(-ef, __uint_as_float(0x3f318000u), v);
    r = fmaf(ef, __uint_as_float(0x395e8083u), r);
    float sq = r * r;
    float fo = sq * sq;
    float p0 = 1.0f + r;
    float p1 = fmaf(__uint_as_float(0x3e2aaaabu), r, 0.5f);         // 1/6
    float p2 = fmaf(__uint_as_float(0x3c088889u), r,                 // 1/120
                    __uint_as_float(0x3d2aaaabu));                   // 1/24
    float p3 = fmaf(__uint_as_float(0x39500d01u), r,                 // 1/5040
                    __uint_as_float(0x3ab60b61u));                   // 1/720
    float low = fmaf(p1, sq, p0);
    float high = fmaf(p3, sq, p2);
    float poly = fmaf(high, fo, low);
    return poly * __int_as_float((e + 127) << 23);
}

// The logistic objectives' gradients of the first `n` rows, the host's
// vector kernel (`simd::logistic_gradient`): `lanes` rows per vector; a
// vector holding a margin above `max_input` in magnitude (or a NaN) takes
// the host's scalar path, which uses the C library's `expf`, so its rows
// get NaN gradients here and the tree grows on the host.
extern "C" __global__ void logistic(const float* __restrict__ margins,
                                    const float* __restrict__ labels,
                                    const float* __restrict__ weights,
                                    int weighted, float scale_pos_weight,
                                    float min_hess, float max_input, u32 lanes,
                                    u64 n, float2* __restrict__ gpair) {
    GRID_STRIDE(i, n) {
        u64 first = i - i % lanes;
        bool regular = true;
        for (u32 k = 0; k < lanes; ++k) {
            regular = regular && fabsf(margins[first + k]) <= max_input;
        }
        if (!regular) {
            gpair[i] = make_float2(__int_as_float(0x7fc00000), __int_as_float(0x7fc00000));
            continue;
        }
        float x = margins[i], y = labels[i];
        float ex = exp_vector(-fabsf(x));
        float den = 1.0f + ex;
        float p = x >= 0.0f ? 1.0f / den : ex / den;
        float w = (weighted ? weights[i] : 1.0f) * (y == 1.0f ? scale_pos_weight : 1.0f);
        float g = (p - y) * w;
        float h = fmaxf(p * (1.0f - p), min_hess) * w;
        gpair[i] = make_float2(g, h);
    }
}

// One value's contribution to a component's exactness statistics (the host
// `SumDomain::of`): the largest magnitude's bits (non-negative floats order
// as their bits), the smallest grain exponent plus 150 (so positive), and
// whether every value is finite. Zeros contribute nothing.
__device__ __forceinline__ void fold_domain(float v, u32& max_bits, u32& grain,
                                            u32& finite) {
    u32 bits = __float_as_uint(v);
    u32 e = (bits >> 23) & 0xffu;
    u32 m = bits & 0x7fffffu;
    if (e == 0xffu) {
        finite = 0;
        return;
    }
    if (e == 0 && m == 0) return;
    u32 mag = bits & 0x7fffffffu;
    max_bits = mag > max_bits ? mag : max_bits;
    int g = e == 0 ? -149 + __ffs(m) - 1 : (int)e - 150 + __ffs(m | 0x800000u) - 1;
    u32 code = (u32)(g + 150);
    grain = code < grain ? code : grain;
}

// Both components' statistics of `n` pairs into `domain` (`[max bits,
// grain + 150, finite]` per component; initialized to `[0, u32::MAX, 1]`).
extern "C" __global__ void grad_domain(const float2* __restrict__ gpair, u64 n,
                                       u32* __restrict__ domain) {
    u32 gm = 0, gg = 0xffffffffu, gf = 1, hm = 0, hg = 0xffffffffu, hf = 1;
    GRID_STRIDE(i, n) {
        float2 p = gpair[i];
        fold_domain(p.x, gm, gg, gf);
        fold_domain(p.y, hm, hg, hf);
    }
    for (int o = 16; o > 0; o >>= 1) {
        gm = max(gm, __shfl_down_sync(FULL_MASK, gm, o));
        gg = min(gg, __shfl_down_sync(FULL_MASK, gg, o));
        gf &= __shfl_down_sync(FULL_MASK, gf, o);
        hm = max(hm, __shfl_down_sync(FULL_MASK, hm, o));
        hg = min(hg, __shfl_down_sync(FULL_MASK, hg, o));
        hf &= __shfl_down_sync(FULL_MASK, hf, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(domain, gm);
        atomicMin(domain + 1, gg);
        atomicAnd(domain + 2, gf);
        atomicMax(domain + 3, hm);
        atomicMin(domain + 4, hg);
        atomicAnd(domain + 5, hf);
    }
}

// `margins[r] += values[leaf]` for every row `r` of every leaf segment,
// one block per 4096-row tile (`leaf << 32 | tile`): each row is in one
// leaf, so every margin receives one `f32` add, the CPU's.
extern "C" __global__ void add_leaves(const u64* __restrict__ segs,
                                      const float* __restrict__ values,
                                      const u64* __restrict__ ptiles,
                                      const u32* __restrict__ rows,
                                      float* __restrict__ margins) {
    u64 code = ptiles[blockIdx.x];
    u32 s = (u32)(code >> 32);
    u64 begin = (u64)(u32)code * 4096u;
    u64 off = segs[2 * s], len = segs[2 * s + 1];
    u64 end = begin + 4096u < len ? begin + 4096u : len;
    float v = values[s];
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
        u32 r = rows[off + i];
        margins[r] = margins[r] + v;
    }
}

// Each `grain`-row chunk's pairs summed as one `f64` chain from zero in row
// order, one thread per chunk (the host's `sum_rows` blocks, for chunks
// whose sums are not exact in integers).
extern "C" __global__ void chunk_chains(const u32* __restrict__ rows, u64 n,
                                        u64 grain, const float2* __restrict__ gpair,
                                        double2* __restrict__ totals) {
    u64 c = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 begin = c * grain;
    if (begin >= n && !(c == 0 && n == 0)) return;
    u64 end = begin + grain < n ? begin + grain : n;
    double g = 0.0, h = 0.0;
    for (u64 i = begin; i < end; ++i) {
        float2 p = gpair[rows[i]];
        g = g + (double)p.x;
        h = h + (double)p.y;
    }
    totals[c] = make_double2(g, h);
}

// The integer totals of the listed rows' grains per `grain`-row chunk, one
// block per chunk (exact when the caller checked the chunks' sums are).
extern "C" __global__ void chunk_totals(const u32* __restrict__ rows, u64 n,
                                        u64 grain,
                                        const longlong2* __restrict__ units,
                                        i64* __restrict__ totals) {
    __shared__ i64 warp_g[32], warp_h[32];
    u64 begin = (u64)blockIdx.x * grain;
    u64 end = begin + grain < n ? begin + grain : n;
    i64 g = 0, h = 0;
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
        longlong2 u = units[rows[i]];
        g += u.x;
        h += u.y;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        g += __shfl_down_sync(FULL_MASK, g, offset);
        h += __shfl_down_sync(FULL_MASK, h, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_g[threadIdx.x >> 5] = g;
        warp_h[threadIdx.x >> 5] = h;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (u32 w = 1; w < (blockDim.x + 31) / 32; ++w) {
            g += warp_g[w];
            h += warp_h[w];
        }
        totals[2 * (u64)blockIdx.x] = g;
        totals[2 * (u64)blockIdx.x + 1] = h;
    }
}

// ---------------------------------------------------------------------------
// Integer histograms

// A 64-bit add into shared memory as two 32-bit atomics with a carry (as
// XGBoost's `AtomicAdd64As32`): each add carries exactly when its own low
// add wrapped, so the words sum to the 64-bit total modulo 2^64.
__device__ __forceinline__ void add_shared(u64* dst, i64 v) {
    u32* p = (u32*)dst;
    u32 lo = (u32)(u64)v;
    u32 hi = (u32)((u64)v >> 32);
    u32 old = atomicAdd(p, lo);
    u32 carry = old > 0xffffffffu - lo ? 1u : 0u;
    u32 add_hi = hi + carry;
    if (add_hi != 0) atomicAdd(p + 1, add_hi);
}

// A unit of histogram work: `count` rows from `rows[begin]`, summed into
// integer slot `target & 0x7fffffff` of the exact accumulators, or (high
// bit set) of the per-chunk partials, which it then owns.
struct Tile {
    u64 begin;
    u32 count;
    u32 target;
};

// A feature group: features `[f0, f1)`, whose global bins `[bin0, bin0 +
// bins)` one block's shared histogram holds.
struct Group {
    u32 f0, f1, bin0, bins;
};

// One block per (tile, group): the tile's rows over the group's features,
// element `idx` = (row `idx / nf`, feature `idx % nf`), so a warp reads one
// row's bins contiguously and its gradient once. A tile's groups are
// consecutive blocks (`blockIdx.x = tile * n_groups + group`), which run
// together, so the tile's rows, bins and gradients come from L2 for all
// but the first group. `SHARED`: privatized in shared memory, then flushed
// (64-bit atomics into an exact accumulator, or plain stores into the
// tile's own partial); otherwise straight into the target with 64-bit
// global atomics (a group too wide for shared memory).
template <typename B, bool SHARED>
__device__ void hist_tile(const B* __restrict__ bins, u32 stride, u32 sentinel,
                          const u32* __restrict__ feature_first,
                          const u32* __restrict__ rows,
                          const Tile* __restrict__ tiles,
                          const Group* __restrict__ groups,
                          const longlong2* __restrict__ units,
                          u64* __restrict__ acc, u64* __restrict__ partials,
                          u64 total_bins, u32 n_groups) {
    extern __shared__ u64 smem[];
    const Tile tile = tiles[blockIdx.x / n_groups];
    const Group g = groups[blockIdx.x % n_groups];
    const bool partial = (tile.target >> 31) != 0;
    const u64 slot = tile.target & 0x7fffffffu;
    u64* target = (partial ? partials : acc) + slot * total_bins * 2;
    if (SHARED) {
        for (u32 i = threadIdx.x; i < 2 * g.bins; i += blockDim.x) smem[i] = 0;
        __syncthreads();
    } else if (partial) {
        for (u32 i = threadIdx.x; i < 2 * g.bins; i += blockDim.x)
            target[2 * (u64)g.bin0 + i] = 0;
        __syncthreads();
    }
    const u32 nf = g.f1 - g.f0;
    const u32 n = tile.count * nf;
    const u32* tile_rows = rows + tile.begin;
    // Four elements per thread per pass, their loads issued before any
    // atomic, so each thread keeps several gathers in flight.
    const u32 step = blockDim.x;
    // Group-relative bins are below 2^31, so this never names one.
    const u32 none = 0xffffffffu;
    for (u32 base = threadIdx.x; base < n; base += 4 * step) {
        u32 rr[4], bb[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            u32 idx = base + k * step;
            bb[k] = none;
            rr[k] = 0;
            if (idx < n) {
                u32 i = idx / nf;
                u32 f = g.f0 + (idx - i * nf);
                u32 r = tile_rows[i];
                u32 b = (u32)bins[(u64)r * stride + f];
                rr[k] = r;
                bb[k] = b == sentinel ? none : feature_first[f] + b - g.bin0;
            }
        }
        longlong2 uu[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            uu[k] = bb[k] != none ? units[rr[k]] : make_longlong2(0, 0);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (bb[k] == none) continue;
            u32 bin = bb[k];
            if (SHARED) {
                add_shared(smem + 2 * bin, uu[k].x);
                add_shared(smem + 2 * bin + 1, uu[k].y);
            } else {
                u64* t = target + 2 * ((u64)g.bin0 + bin);
                atomicAdd(t, (u64)uu[k].x);
                atomicAdd(t + 1, (u64)uu[k].y);
            }
        }
    }
    if (!SHARED) return;
    __syncthreads();
    for (u32 b = threadIdx.x; b < g.bins; b += blockDim.x) {
        u64 x = smem[2 * b], y = smem[2 * b + 1];
        u64* t = target + 2 * ((u64)g.bin0 + b);
        if (partial) {
            t[0] = x;
            t[1] = y;
        } else {
            if (x != 0) atomicAdd(t, x);
            if (y != 0) atomicAdd(t + 1, y);
        }
    }
}

// Integer histograms of `segs` chunks of `seg_rows` rows each, as `f64`
// chains: one thread per (chunk, feature) adds the chunk's rows in order
// to its own feature's bins of the chunk's `f64` partial (zeroed), so every
// bin is a chain from `+0.0` in row order, the CPU's.
template <typename B>
__device__ void hist_chain(const B* __restrict__ bins, u32 stride, u32 n_cols,
                           u32 sentinel, const u32* __restrict__ feature_first,
                           const u32* __restrict__ rows, u64 n, u64 seg_rows,
                           u64 segs, const float2* __restrict__ gpair,
                           double2* __restrict__ partials, u64 total_bins) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= segs * n_cols) return;
    u64 seg = t / n_cols;
    u32 f = (u32)(t - seg * n_cols);
    u64 begin = seg * seg_rows;
    u64 end = begin + seg_rows < n ? begin + seg_rows : n;
    double2* h = partials + seg * total_bins + feature_first[f];
    for (u64 i = begin; i < end; ++i) {
        u32 r = rows[i];
        u32 b = (u32)bins[(u64)r * stride + f];
        if (b == sentinel) continue;
        float2 p = gpair[r];
        double2 a = h[b];
        a.x = a.x + (double)p.x;
        a.y = a.y + (double)p.y;
        h[b] = a;
    }
}

// ---------------------------------------------------------------------------
// Partition

// How a split routes a row: feature, then present feature-local bins below
// `limit` go left (or, with flag 2, `table[table_at + bin]`), a missing
// value follows flag 1 (`default_left`).
struct Rule {
    u32 feature, limit, table_at, flags;
};

// A partition tile: rows `[k * PART_TILE, ...)` of split `s`'s segment,
// encoded `s << 32 | k`.
#define PART_TILE 4096u

// Each tile's rows' directions (`flags[off + i]`, 1 = left) from the
// feature-major bins (`cols[f * n_rows + r]`: one byte stream per feature,
// so a node's ascending rows read few sectors), and the tile's left count.
template <typename B>
__device__ void route_count(const B* __restrict__ cols, u64 n_rows, u32 sentinel,
                            const u64* __restrict__ segs,
                            const Rule* __restrict__ rules,
                            const u8* __restrict__ table,
                            const u64* __restrict__ ptiles,
                            const u32* __restrict__ rows,
                            u8* __restrict__ flags,
                            u32* __restrict__ tile_left) {
    __shared__ u32 warp_sum[32];
    u64 code = ptiles[blockIdx.x];
    u32 s = (u32)(code >> 32);
    u64 begin = (u64)(u32)code * PART_TILE;
    u64 off = segs[2 * s], len = segs[2 * s + 1];
    u64 end = begin + PART_TILE < len ? begin + PART_TILE : len;
    Rule rule = rules[s];
    const B* col = cols + (u64)rule.feature * n_rows;
    u32 count = 0;
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
        u32 b = (u32)col[rows[off + i]];
        bool left = b == sentinel ? (rule.flags & 1u) != 0
                    : (rule.flags & 2u) ? table[rule.table_at + b] != 0
                                        : b < rule.limit;
        flags[off + i] = left ? 1 : 0;
        count += left ? 1u : 0u;
    }
    for (int o = 16; o > 0; o >>= 1) count += __shfl_down_sync(FULL_MASK, count, o);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        u32 total = 0;
        for (u32 w = 0; w < (blockDim.x + 31) / 32; ++w) total += warp_sum[w];
        tile_left[blockIdx.x] = total;
    }
}

// Per split (one thread each): the exclusive prefix of its tiles' left
// counts, in tile order, and its left total.
extern "C" __global__ void route_scan(const uint2* __restrict__ split_tiles,
                                      u32 n_splits, u32* __restrict__ tile_left,
                                      u32* __restrict__ left_len) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_splits) return;
    uint2 t = split_tiles[s];
    u32 run = 0;
    for (u32 k = t.x; k < t.x + t.y; ++k) {
        u32 c = tile_left[k];
        tile_left[k] = run;
        run += c;
    }
    left_len[s] = run;
}

// Stable scatter of each tile's rows into `scratch` by their flags: left
// rows to the segment's front in row order, right rows after all left
// ones. Rows are ranked a block-width round at a time with warp ballots.
extern "C" __global__ void route_scatter(const u64* __restrict__ segs,
                                         const u64* __restrict__ ptiles,
                                         const u32* __restrict__ rows,
                                         const u8* __restrict__ flags,
                                         const u32* __restrict__ tile_left,
                                         const u32* __restrict__ left_len,
                                         u32* __restrict__ scratch) {
    __shared__ u32 warp_left[32];
    __shared__ u32 warp_valid[32];
    u64 code = ptiles[blockIdx.x];
    u32 s = (u32)(code >> 32);
    u64 begin = (u64)(u32)code * PART_TILE;
    u64 off = segs[2 * s], len = segs[2 * s + 1];
    u64 end = begin + PART_TILE < len ? begin + PART_TILE : len;
    u64 left_at = off + tile_left[blockIdx.x];
    u64 right_at = off + left_len[s] + (begin - tile_left[blockIdx.x]);
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const u32 warps = (blockDim.x + 31) / 32;
    const u32 below = (1u << lane) - 1u;
    for (u64 base = begin; base < end; base += blockDim.x) {
        u64 i = base + threadIdx.x;
        bool valid = i < end;
        u32 r = valid ? rows[off + i] : 0;
        bool left = valid && flags[off + i] != 0;
        u32 lmask = __ballot_sync(FULL_MASK, left);
        u32 vmask = __ballot_sync(FULL_MASK, valid);
        if (lane == 0) {
            warp_left[warp] = __popc(lmask);
            warp_valid[warp] = __popc(vmask);
        }
        __syncthreads();
        u32 lbefore = 0, rbefore = 0, ltotal = 0, rtotal = 0;
        for (u32 w = 0; w < warps; ++w) {
            u32 l = warp_left[w], v = warp_valid[w];
            if (w < warp) {
                lbefore += l;
                rbefore += v - l;
            }
            ltotal += l;
            rtotal += v - l;
        }
        if (left) {
            scratch[left_at + lbefore + __popc(lmask & below)] = r;
        } else if (valid) {
            scratch[right_at + rbefore + __popc(vmask & ~lmask & below)] = r;
        }
        left_at += ltotal;
        right_at += rtotal;
        __syncthreads();
    }
}

// Copy each tile's span of `scratch` back to `rows`.
extern "C" __global__ void route_copy(const u64* __restrict__ segs,
                                      const u64* __restrict__ ptiles,
                                      const u32* __restrict__ scratch,
                                      u32* __restrict__ rows) {
    u64 code = ptiles[blockIdx.x];
    u32 s = (u32)(code >> 32);
    u64 begin = (u64)(u32)code * PART_TILE;
    u64 off = segs[2 * s], len = segs[2 * s + 1];
    u64 end = begin + PART_TILE < len ? begin + PART_TILE : len;
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x)
        rows[off + i] = scratch[off + i];
}

// ---------------------------------------------------------------------------
// Reductions into the `f64` histograms

// Exact accumulators to `f64`: slot `nodes[k].x` into output slot
// `nodes[k].y`. `(f64)K * 2^grain` is exact (|K| <= 2^53), and equals the
// CPU's sum, which is exact too.
extern "C" __global__ void finalize_exact(const u64* __restrict__ acc,
                                          const uint2* __restrict__ nodes,
                                          u64 total_bins, double grad_value,
                                          double hess_value,
                                          double2* __restrict__ out) {
    uint2 nd = nodes[blockIdx.y];
    for (u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x; b < total_bins;
         b += (u64)gridDim.x * blockDim.x) {
        const u64* a = acc + ((u64)nd.x * total_bins + b) * 2;
        out[(u64)nd.y * total_bins + b] =
            make_double2((double)(i64)a[0] * grad_value, (double)(i64)a[1] * hess_value);
    }
}

// Integer chunk partials to `f64`, per bin in chunk order: node `k` reads
// partial slots `[x, x + y)` into output slot `z`, copying the first when
// `w` (its first chunk) and adding the rest, the CPU's copy-then-add.
extern "C" __global__ void reduce_chunks(const u64* __restrict__ partials,
                                         const uint4* __restrict__ nodes,
                                         u64 total_bins, double grad_value,
                                         double hess_value,
                                         double2* __restrict__ out) {
    uint4 nd = nodes[blockIdx.y];
    for (u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x; b < total_bins;
         b += (u64)gridDim.x * blockDim.x) {
        double2* o = out + (u64)nd.z * total_bins + b;
        double g, h;
        u32 s = 0;
        if (nd.w) {
            const u64* p = partials + ((u64)nd.x * total_bins + b) * 2;
            g = (double)(i64)p[0] * grad_value;
            h = (double)(i64)p[1] * hess_value;
            s = 1;
        } else {
            g = o->x;
            h = o->y;
        }
        for (; s < nd.y; ++s) {
            const u64* p = partials + ((u64)(nd.x + s) * total_bins + b) * 2;
            g = g + (double)(i64)p[0] * grad_value;
            h = h + (double)(i64)p[1] * hess_value;
        }
        *o = make_double2(g, h);
    }
}

// `f64` chain partials of `segs` chunks into `out`, per bin in chunk order
// (the first copied when `init`).
extern "C" __global__ void reduce_chains(const double2* __restrict__ partials,
                                         u64 segs, u64 total_bins, int init,
                                         double2* __restrict__ out) {
    GRID_STRIDE(b, total_bins) {
        double2 a;
        u64 s = 0;
        if (init) {
            a = partials[b];
            s = 1;
        } else {
            a = out[b];
        }
        for (; s < segs; ++s) {
            double2 p = partials[s * total_bins + b];
            a.x = a.x + p.x;
            a.y = a.y + p.y;
        }
        out[b] = a;
    }
}

// ---------------------------------------------------------------------------
// Resident split search: histograms stay in device slots.

// Each `(parent, built)` slot pair's parent becomes `parent - built`, the
// host's `subtract_in_place`.
extern "C" __global__ void subtract_hists(double2* __restrict__ pool,
                                          const u32* __restrict__ pairs, u64 n_pairs,
                                          u64 total_bins) {
    GRID_STRIDE(i, n_pairs * total_bins) {
        u64 p = i / total_bins, b = i % total_bins;
        double2* parent = pool + (u64)pairs[2 * p] * total_bins + b;
        double2 c = pool[(u64)pairs[2 * p + 1] * total_bins + b];
        double2 a = *parent;
        *parent = make_double2(a.x - c.x, a.y - c.y);
    }
}

// The scorer's child weight (`SplitScorer::score_run`'s `weight`): the
// soft-thresholded gradient over `H + lambda` in `f64`, `max_delta_step`,
// rounded to `f32`, then clamped to the node's bounds.
__device__ __forceinline__ float scan_weight(double g, double h, double lambda,
                                             double alpha, double max_delta_step,
                                             float lower, float upper) {
    double t = g > alpha ? g - alpha : (g < -alpha ? g + alpha : 0.0);
    double w = -t / (h + lambda);
    if (max_delta_step != 0.0 && fabs(w) > max_delta_step) w = copysign(max_delta_step, w);
    float wf = __double2float_rn(w);
    return wf < lower ? lower : (wf > upper ? upper : wf);
}

// XGBoost's `CalcGainGivenWeight` at an `f32` weight (`w * w` in `f32`).
__device__ __forceinline__ double scan_gain(double g, double h, float w, double lambda,
                                            double alpha) {
    return -(2.0 * g * (double)w + (h + lambda) * (double)(w * w) +
             2.0 * alpha * (double)fabsf(w));
}

struct ScanReg {
    double lambda, alpha, max_delta_step, min_child_weight;
    float root_gain, lower, upper;
    int dir;
};

// One candidate's loss change, `-inf` when a child is invalid or the
// monotone direction is violated (`SplitScorer::score_run`).
__device__ __forceinline__ float scan_score(double lg, double lh, double rg, double rh,
                                            const ScanReg& r) {
    bool valid = lh > 0.0 && rh > 0.0 && lh >= r.min_child_weight && rh >= r.min_child_weight;
    float wl = scan_weight(lg, lh, r.lambda, r.alpha, r.max_delta_step, r.lower, r.upper);
    float wr = scan_weight(rg, rh, r.lambda, r.alpha, r.max_delta_step, r.lower, r.upper);
    bool monotone = r.dir > 0 ? wl <= wr : (r.dir < 0 ? wl >= wr : true);
    float chg = (__double2float_rn(scan_gain(lg, lh, wl, r.lambda, r.alpha)) +
                 __double2float_rn(scan_gain(rg, rh, wr, r.lambda, r.alpha))) -
                r.root_gain;
    return valid && monotone ? chg : -__int_as_float(0x7f800000);
}

// Warps per `scan_splits` block (`SCAN_WARPS` in `mod.rs`).
#define SCAN_WARPS 4

// One warp per task `(request, feature, slot, dir)`: the feature's numeric
// scan in `scan_batched`'s order (the forward pass, then the backward one
// when the feature has missing values in the node), keeping the first
// candidate with the largest finite loss change. The prefix sums must be
// the host's sequential chains, so lane 0 forms them 32 bins at a time
// into shared memory; the lanes then score those candidates in parallel
// (the costly part: two `f64` divisions each), and a warp reduction keeps
// the largest loss at the smallest position, which is what the host's
// strict `>` over the candidate sequence keeps. `meta` gets `(status: 0
// empty, 1 best, 2 NaN; bin offset; loss bits; backward)`, `acc` the
// winning pass's accumulated statistics.
extern "C" __global__ void __launch_bounds__(32 * SCAN_WARPS)
    scan_splits(const double2* __restrict__ pool, const u32* __restrict__ feature_first,
                u64 total_bins, const u32* __restrict__ tasks, u64 n_tasks,
                const double2* __restrict__ totals, const float* __restrict__ params,
                double lambda, double alpha, double max_delta_step, double min_child_weight,
                int dense, u32* __restrict__ meta, double2* __restrict__ acc) {
    __shared__ double2 chain[SCAN_WARPS][32];
    const u32 lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const u64 warps = (u64)gridDim.x * SCAN_WARPS;
    for (u64 t = (u64)blockIdx.x * SCAN_WARPS + w; t < n_tasks; t += warps) {
        u32 request = tasks[4 * t], f = tasks[4 * t + 1], slot = tasks[4 * t + 2];
        ScanReg r;
        r.lambda = lambda;
        r.alpha = alpha;
        r.max_delta_step = max_delta_step;
        r.min_child_weight = min_child_weight;
        r.root_gain = params[3 * request];
        r.lower = params[3 * request + 1];
        r.upper = params[3 * request + 2];
        r.dir = (int)tasks[4 * t + 3];
        const double2 total = totals[request];
        const u32 first = feature_first[f], len = feature_first[f + 1] - first;
        const double2* bins = pool + (u64)slot * total_bins + first;
        float best = -__int_as_float(0x7f800000);
        u32 pos = 0xffffffffu;
        double best_g = 0.0, best_h = 0.0;
        bool nan = false;
        // Lane 0's running chain of the current pass.
        double g = 0.0, h = 0.0;
        for (int backward = 0; backward < 2; ++backward) {
            if (backward) {
                // The forward pass's final sums decide whether the
                // feature has missing values here.
                double fg = __shfl_sync(FULL_MASK, g, 0), fh = __shfl_sync(FULL_MASK, h, 0);
                if (dense || (fg == total.x && fh == total.y)) break;
                g = 0.0;
                h = 0.0;
            }
            for (u32 base = 0; base < len; base += 32) {
                const u32 n = len - base < 32 ? len - base : 32;
                if (lane == 0) {
                    for (u32 i = 0; i < n; ++i) {
                        double2 b = bins[backward ? len - 1 - (base + i) : base + i];
                        g = g + b.x;
                        h = h + b.y;
                        chain[w][i] = make_double2(g, h);
                    }
                }
                __syncwarp();
                if (lane < n) {
                    const double2 a = chain[w][lane];
                    const float l = backward
                                        ? scan_score(total.x - a.x, total.y - a.y, a.x, a.y, r)
                                        : scan_score(a.x, a.y, total.x - a.x, total.y - a.y, r);
                    if (isnan(l)) {
                        nan = true;
                    } else if (l > best && isfinite(l)) {
                        best = l;
                        pos = (backward ? len : 0) + base + lane;
                        best_g = a.x;
                        best_h = a.y;
                    }
                }
                __syncwarp();
            }
        }
        // First maximum: the larger loss, then the smaller position (the
        // forward pass's `k`, or `len + k` for the backward pass's step `k`).
        for (int o = 16; o > 0; o >>= 1) {
            float ob = __shfl_down_sync(FULL_MASK, best, o);
            u32 op = __shfl_down_sync(FULL_MASK, pos, o);
            double og = __shfl_down_sync(FULL_MASK, best_g, o);
            double oh = __shfl_down_sync(FULL_MASK, best_h, o);
            if (ob > best || (ob == best && op < pos)) {
                best = ob;
                pos = op;
                best_g = og;
                best_h = oh;
            }
        }
        nan = __any_sync(FULL_MASK, nan);
        if (lane == 0) {
            const bool found = pos != 0xffffffffu;
            const bool back = found && pos >= len;
            meta[4 * t] = nan ? 2u : (found ? 1u : 0u);
            // A backward step `k` puts bins `>= len - 1 - k` right.
            meta[4 * t + 1] = found ? (back ? 2 * len - 1 - pos : pos) : 0u;
            meta[4 * t + 2] = __float_as_uint(best);
            meta[4 * t + 3] = back ? 1u : 0u;
            acc[t] = make_double2(best_g, best_h);
        }
    }
}

// ---------------------------------------------------------------------------
// Instantiations per bin width

#define HIST_THREADS 512

#define INSTANTIATE(B, W)                                                      \
    extern "C" __global__ void __launch_bounds__(HIST_THREADS)                 \
        hist_shared_##W(const B* bins, u32 stride, u32 sentinel,               \
                        const u32* feature_first, const u32* rows,             \
                        const Tile* tiles, const Group* groups,                \
                        const longlong2* units, u64* acc, u64* partials,       \
                        u64 total_bins, u32 n_groups) {                        \
        hist_tile<B, true>(bins, stride, sentinel, feature_first, rows, tiles, \
                           groups, units, acc, partials, total_bins,           \
                           n_groups);                                          \
    }                                                                          \
    extern "C" __global__ void __launch_bounds__(HIST_THREADS)                 \
        hist_global_##W(const B* bins, u32 stride, u32 sentinel,               \
                        const u32* feature_first, const u32* rows,             \
                        const Tile* tiles, const Group* groups,                \
                        const longlong2* units, u64* acc, u64* partials,       \
                        u64 total_bins, u32 n_groups) {                        \
        hist_tile<B, false>(bins, stride, sentinel, feature_first, rows,       \
                            tiles, groups, units, acc, partials, total_bins,   \
                            n_groups);                                         \
    }                                                                          \
    extern "C" __global__ void hist_chain_##W(                                 \
        const B* bins, u32 stride, u32 n_cols, u32 sentinel,                   \
        const u32* feature_first, const u32* rows, u64 n, u64 seg_rows,        \
        u64 segs, const float2* gpair, double2* partials, u64 total_bins) {    \
        hist_chain<B>(bins, stride, n_cols, sentinel, feature_first, rows, n,  \
                      seg_rows, segs, gpair, partials, total_bins);            \
    }                                                                          \
    extern "C" __global__ void route_count_##W(                                \
        const B* cols, u64 n_rows, u32 sentinel, const u64* segs,              \
        const Rule* rules, const u8* table, const u64* ptiles,                 \
        const u32* rows, u8* flags, u32* tile_left) {                          \
        route_count<B>(cols, n_rows, sentinel, segs, rules, table, ptiles,     \
                       rows, flags, tile_left);                                \
    }

INSTANTIATE(u8, u8)
INSTANTIATE(u16, u16)
INSTANTIATE(u32, u32)

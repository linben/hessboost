//! NVIDIA CUDA acceleration for Linux (opt-in `cuda` feature).
//!
//! **Training** (`device = cuda`, see
//! [`TrainingParams::device`](crate::config::TrainingParams::device)): the
//! trained model is bit-identical to CPU training (at any thread count),
//! and the tree's rows live on the GPU. Depthwise growth handles a whole
//! level per round trip: the GPU partitions every splitting node (a stable
//! partition, so each child keeps its rows in ascending order) and builds
//! the smaller child of every split. On numeric features the histograms
//! then stay on the GPU: it derives each sibling by subtraction and scans
//! every feature's split candidates in the CPU's order and arithmetic, and
//! the host merges the per-feature winners with XGBoost's tie rule (a node
//! whose scan scores a NaN is searched on the host from its histogram).
//! With categorical features the histograms are downloaded and searched on
//! the host, as the CPU builder does; loss-guided growth does the same one
//! expansion at a time. Symmetric (`grow_policy = symmetric`) trees build
//! their histograms here node by node, with the rows uploaded per node.
//!
//! For `reg:squarederror` and the logistic objectives on one label column
//! in a plain `gbtree` (no row sampling, DART, linear leaves, SGLB, model
//! shrinkage, or reuse penalties), whole rounds run on the GPU: it keeps
//! the training margins, computes the gradients from them with the host's
//! operations (the logistic ones are the host's vector kernel, so they need
//! AVX2/FMA or NEON; the few trailing rows the host runs scalar are
//! computed on the host), and adds each leaf's value to its rows. A round
//! the GPU cannot reproduce (non-finite gradients, logistic margins beyond
//! the vector kernel's ±80) runs on the host. Other configurations compute
//! the gradients on the host each round, upload them once per tree, and
//! read the leaf rows back once per tree to update the margins.
//!
//! # Requirements
//!
//! - Linux with an NVIDIA GPU and a driver supporting CUDA 12.8 or later.
//! - NVRTC (`libnvrtc`), loadable as `libnvrtc.so` or `libnvrtc.so.12`:
//!   the CUDA toolkit's `lib64` directory on the loader path (CUDA 13
//!   toolkits ship `libnvrtc.so` there), or `LD_LIBRARY_PATH` pointing at it.
//!   The kernels are compiled once per process, for the device's own
//!   architecture, straight to machine code (CUBIN), so no PTX JIT runs and
//!   a newer NVRTC than the driver is not a problem.
//! - Nothing at build time: both libraries are opened at run time, and
//!   their absence makes the backend unavailable rather than failing to
//!   load the crate.
//!
//! # Exactness
//!
//! The CPU adds each histogram bin in `f64` in a fixed order: one chain in
//! row order, or fixed chunks of rows each chained from zero and then added
//! in chunk order (`tree::hist::sum_order`, chosen from the index and the
//! node's rows alone). Every node uses the first strategy that applies:
//!
//! 1. **Exact integers.** When every sum of the node's rows is exact
//!    (`n * max <= 2^53` gradient grains for both components, the domain
//!    the Metal backend also uses; proof in the private `exact_sum` module),
//!    the GPU sums 64-bit grain counts in any order (shared-memory
//!    histograms per row tile and feature group, flushed with 64-bit
//!    atomics) and scales them back exactly.
//! 2. **Exact chunks.** For a chunked node whose *chunks* are exact, each
//!    chunk's integer sum is that chunk's `f64` chain, and the GPU then adds
//!    the chunk partials in chunk order in `f64`, the CPU's own operations.
//! 3. **Chains.** Otherwise, for a chunked node or a node below 8,192 rows,
//!    one GPU thread per (chunk, feature) runs that feature's `f64` chain in
//!    row order, and the chunk partials are added in chunk order.
//! 4. **CPU.** A single-chain node of 8,192 rows or more outside the exact
//!    domain (a dense index's unsampled root, or any node of a dense index
//!    of at most 2^18 rows) runs the CPU backend's build.
//!
//! The root's statistics come from the GPU's exact integer total when the
//! root's sums are exact, else from the host's row-order chain. The kernels
//! are compiled without FP contraction, flush-to-zero, or approximate
//! division, so every `f64` operation is the single IEEE operation the CPU
//! performs; there are no floating-point atomics. Trees with a non-finite
//! gradient or Hessian (NaN payloads differ between CPU and GPU
//! arithmetic) build every node on the CPU. A CUDA error is sticky (the
//! context is unusable afterwards): the tree in progress is regrown on the
//! host, and every later tree too, so the result is unchanged.
//!
//! # Limitations
//!
//! - Training only; prediction stays on the CPU.
//! - Refused with `device = cuda`: `tree_method = exact`/`approx`,
//!   `use_quantized_grad`, `gblinear`, `process_type = update`,
//!   `multi_strategy = multi_output_tree`, online updates, and budget mode.
//!   Trees with reuse penalties (`toad_penalty_*`) grow on the host, with
//!   per-node GPU histograms.
//! - Resident on the device: the binned index as row-major feature-local
//!   bins (1, 2 or 4 bytes per row and feature), the gradients (24 bytes
//!   per row), three row buffers (12 bytes per row), the margins, labels
//!   and weights of device-side rounds (12 bytes per row), and the
//!   histograms of one level (resident growth: of every open node, at most
//!   `2^(max_depth - 1)` of them, when that fits in half the free memory).

pub(crate) mod compile;

use crate::backend::exact_sum::SumDomain;
use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{
    CpuBackend, DeviceLoss, FeatureScan, HistSlot, Histogram, HistogramBackend, Partitioned,
    RowEngine, RowRule, RowSplit, ScanRequest, Segment, SumOrder, sum_order, zeroed,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits, sys,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Threads per block of the elementwise kernels.
const THREADS: u32 = 256;
/// Grid-stride kernels launch at most this many blocks per SM.
const BLOCKS_PER_SM: u32 = 8;
/// Threads per histogram block (`HIST_THREADS` in `kernels.cu`). Smaller
/// blocks than XGBoost's 1024 keep several resident per SM, so one block's
/// shared-memory flush overlaps another's gathers (measured on an L40S).
const HIST_THREADS: u32 = 512;
/// Rows per partition tile (`PART_TILE` in `kernels.cu`).
const PART_TILE: usize = 4096;
/// Threads per partition block.
const PART_THREADS: u32 = 512;
/// Warps (one per scanned feature) per split-scan block (`SCAN_WARPS` in
/// `kernels.cu`).
const SCAN_WARPS: usize = 4;
/// Rows per histogram tile of an exact node.
const HIST_TILE: usize = 4096;
/// Largest shared histogram per block; XGBoost's single-target limit.
const MAX_SHARED_BYTES: usize = 96 << 10;
/// Shared memory the driver reserves per resident block.
const RESERVED_SHARED_BYTES: usize = 1 << 10;
/// Bytes of per-chunk partial histograms held at once. A level with more
/// chunks is built in waves of chunks, each reduced into the output in
/// chunk order before the next, as the CPU's waves are.
const PARTIAL_BYTES: usize = 512 << 20;
/// The oldest driver the backend runs on (`cuDriverGetVersion` encoding):
/// the API version the bindings are built against, CUDA 12.8.
const MIN_DRIVER: i32 = 12_080;
/// Most nodes one reduction launch covers (a grid's `y` limit).
const MAX_GRID_Y: usize = 65_535;

/// Whether CUDA device 0 is available: the driver and NVRTC load, the
/// device exists, and the kernels compile for it.
#[must_use]
pub fn available() -> bool {
    device(0).is_ok()
}

/// Why CUDA device 0 is unavailable (no driver, no NVRTC, no device, or a
/// kernel compile failure), for diagnostics; `None` when it is available.
#[must_use]
pub fn unavailable_reason() -> Option<String> {
    device(0).err()
}

/// The name of CUDA device 0, if it is available (for diagnostics and
/// benchmarks).
#[must_use]
pub fn device_name() -> Option<String> {
    device(0).ok().map(|device| device.name.clone())
}

/// The kernels of one device's module; per-width kernels are indexed by
/// [`DeviceBins::width`] (`u8`, `u16`, `u32`).
struct Kernels {
    stage_units: CudaFunction,
    iota_rows: CudaFunction,
    chunk_totals: CudaFunction,
    hist_shared: [CudaFunction; 3],
    hist_global: [CudaFunction; 3],
    hist_chain: [CudaFunction; 3],
    route_count: [CudaFunction; 3],
    route_scatter: CudaFunction,
    route_scan: CudaFunction,
    route_copy: CudaFunction,
    finalize_exact: CudaFunction,
    reduce_chunks: CudaFunction,
    reduce_chains: CudaFunction,
    squared_error: CudaFunction,
    logistic: CudaFunction,
    grad_domain: CudaFunction,
    add_leaves: CudaFunction,
    chunk_chains: CudaFunction,
    subtract_hists: CudaFunction,
    scan_splits: CudaFunction,
}

/// One opened CUDA device: the stream every backend on it uses (holding
/// its primary context), its compiled kernels, and its limits.
struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    name: String,
    sm_count: u32,
    /// Dynamic shared memory one histogram block may use.
    shared_bytes: usize,
    /// Set by the first CUDA error: the context is unusable afterwards, so
    /// every later build on this device runs on the CPU.
    failed: AtomicBool,
}

impl Device {
    fn open(ordinal: usize) -> std::result::Result<Self, String> {
        libraries()?;
        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NO_DEVICE => 0,
            Err(e) => return Err(format!("CUDA init failed: {e}")),
        };
        let count = usize::try_from(count).unwrap_or(0);
        if ordinal >= count {
            return Err(format!("no CUDA device {ordinal} ({count} found)"));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CUDA context: {e}"))?;
        // SAFETY: called before this context allocates any slice, and every
        // slice of it is used on the one stream below only, so no slice
        // needs cross-stream event tracking.
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| format!("CUDA compute capability: {e}"))?;
        let arch = format!("sm_{major}{minor}");
        let module = compile::load(&ctx, &arch)?;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("CUDA kernel `{name}`: {e}"))
        };
        let widths = |prefix: &str| -> std::result::Result<[CudaFunction; 3], String> {
            Ok([
                function(&format!("{prefix}_u8"))?,
                function(&format!("{prefix}_u16"))?,
                function(&format!("{prefix}_u32"))?,
            ])
        };
        let kernels = Kernels {
            stage_units: function("stage_units")?,
            iota_rows: function("iota_rows")?,
            chunk_totals: function("chunk_totals")?,
            hist_shared: widths("hist_shared")?,
            hist_global: widths("hist_global")?,
            hist_chain: widths("hist_chain")?,
            route_count: widths("route_count")?,
            route_scatter: function("route_scatter")?,
            route_scan: function("route_scan")?,
            route_copy: function("route_copy")?,
            finalize_exact: function("finalize_exact")?,
            reduce_chunks: function("reduce_chunks")?,
            reduce_chains: function("reduce_chains")?,
            squared_error: function("squared_error")?,
            logistic: function("logistic")?,
            grad_domain: function("grad_domain")?,
            add_leaves: function("add_leaves")?,
            chunk_chains: function("chunk_chains")?,
            subtract_hists: function("subtract_hists")?,
            scan_splits: function("scan_splits")?,
        };
        let attribute = |attribute, what: &str| {
            ctx.attribute(attribute)
                .map_err(|e| format!("CUDA {what}: {e}"))
        };
        let sm_count = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "SM count",
        )?;
        let optin = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            "shared memory limit",
        )?;
        let sm_shared = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            "SM shared memory",
        )?;
        let sm_threads = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            "SM thread limit",
        )?;
        // As many histogram blocks as the SM's threads allow share its
        // shared memory (Ada: three of 32 KiB).
        let resident = usize::try_from(sm_threads).unwrap_or(0) / HIST_THREADS as usize;
        let per_block = usize::try_from(sm_shared).unwrap_or(0) / resident.max(1);
        let shared_bytes = per_block
            .saturating_sub(RESERVED_SHARED_BYTES)
            .min(usize::try_from(optin).unwrap_or(0))
            .clamp(16 << 10, MAX_SHARED_BYTES);
        for kernel in &kernels.hist_shared {
            // SAFETY: a valid function of the loaded module; the attribute
            // only raises the dynamic shared-memory cap to the device's
            // opt-in limit (or less).
            unsafe {
                sys::cuFuncSetAttribute(
                    kernel.cu_function(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_bytes as i32,
                )
            }
            .result()
            .map_err(|e| format!("CUDA shared memory attribute: {e}"))?;
        }
        let stream = ctx.new_stream().map_err(|e| format!("CUDA stream: {e}"))?;
        let name = ctx.name().map_err(|e| format!("CUDA device name: {e}"))?;
        Ok(Device {
            stream,
            kernels,
            name,
            sm_count: u32::try_from(sm_count).unwrap_or(1).max(1),
            shared_bytes,
            failed: AtomicBool::new(false),
        })
    }

    /// A launch shape for a grid-stride kernel over `work` items.
    fn grid(&self, work: usize) -> LaunchConfig {
        let blocks = work.div_ceil(THREADS as usize).max(1);
        let cap = (self.sm_count * BLOCKS_PER_SM) as usize;
        LaunchConfig {
            grid_dim: (blocks.min(cap) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// A launch shape with one thread per item (no grid stride).
    fn one_per(work: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (work.div_ceil(THREADS as usize).max(1) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// A grid-stride launch over `total_bins` bins for each of `nodes`
    /// nodes (the grid's `y`).
    fn per_node_bins(total_bins: usize, nodes: usize) -> LaunchConfig {
        let blocks = total_bins.div_ceil(THREADS as usize).clamp(1, 64);
        LaunchConfig {
            grid_dim: (blocks as u32, nodes as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The driver and NVRTC both load, and the driver is new enough. Checked
/// before any other `cudarc` call: its lazy loaders panic when a library
/// is missing, and the release profile aborts on panic.
fn libraries() -> std::result::Result<(), String> {
    // SAFETY: only tries to open the shared libraries by name.
    if !unsafe { sys::is_culib_present() } {
        return Err("libcuda not found (no NVIDIA driver is installed)".into());
    }
    // SAFETY: as above.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(
            "libnvrtc not found: install the CUDA toolkit's NVRTC and put the \
                    directory holding `libnvrtc.so` on the loader path"
                .into(),
        );
    }
    let mut version = 0;
    // SAFETY: the driver library loads (checked above), and the call only
    // writes the version through the pointer.
    unsafe { sys::cuDriverGetVersion(&raw mut version) }
        .result()
        .map_err(|e| format!("CUDA driver version: {e}"))?;
    if version < MIN_DRIVER {
        return Err(format!(
            "the NVIDIA driver supports CUDA {}.{}; the backend needs 12.8 or later",
            version / 1000,
            version % 1000 / 10
        ));
    }
    Ok(())
}

/// The opened devices, by ordinal (each opened once per process; a failure
/// is remembered too).
static DEVICES: Mutex<Vec<(usize, Opened)>> = Mutex::new(Vec::new());

/// A device opened once, or why it could not be.
type Opened = std::result::Result<Arc<Device>, String>;

/// Open (once) CUDA device `ordinal`.
fn device(ordinal: usize) -> std::result::Result<Arc<Device>, String> {
    let mut devices = DEVICES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, opened)) = devices.iter().find(|(o, _)| *o == ordinal) {
        return opened.clone();
    }
    let opened = Device::open(ordinal).map(Arc::new);
    devices.push((ordinal, opened.clone()));
    opened
}

/// How many nodes (and rows) a [`CudaHistBackend`] built with each
/// strategy of the [module docs](self), for benchmarks and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NodeCounts {
    /// Nodes summed as exact integers in one piece (strategy 1).
    pub exact_nodes: u64,
    /// Nodes summed as exact integer chunks reduced in `f64` (strategy 2).
    pub exact_chunk_nodes: u64,
    /// Nodes summed as `f64` chains on the GPU (strategy 3).
    pub chain_nodes: u64,
    /// Nodes built by the CPU backend (strategy 4, input mismatches, and
    /// every node after a CUDA error).
    pub cpu_nodes: u64,
    /// Rows of the nodes counted in `exact_nodes`.
    pub exact_rows: u64,
    /// Rows of the nodes counted in `exact_chunk_nodes`.
    pub exact_chunk_rows: u64,
    /// Rows of the nodes counted in `chain_nodes`.
    pub chain_rows: u64,
    /// Rows of the nodes counted in `cpu_nodes`.
    pub cpu_rows: u64,
}

/// Atomic [`NodeCounts`].
#[derive(Default)]
struct Counters([AtomicU64; 8]);

impl Counters {
    fn count(&self, strategy: Strategy, rows: usize) {
        let k = strategy as usize;
        self.0[k].fetch_add(1, Ordering::Relaxed);
        self.0[k + 4].fetch_add(rows as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> NodeCounts {
        let v = |i: usize| self.0[i].load(Ordering::Relaxed);
        NodeCounts {
            exact_nodes: v(0),
            exact_chunk_nodes: v(1),
            chain_nodes: v(2),
            cpu_nodes: v(3),
            exact_rows: v(4),
            exact_chunk_rows: v(5),
            chain_rows: v(6),
            cpu_rows: v(7),
        }
    }
}

/// A node's strategy (the [module docs](self)' numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Exact = 0,
    ExactChunks = 1,
    Chains = 2,
    Cpu = 3,
}

/// The device copy of the binned index: row-major, `n_cols` feature-local
/// bins per row.
enum DeviceBins {
    U8(CudaSlice<u8>),
    U16(CudaSlice<u16>),
    U32(CudaSlice<u32>),
}

impl DeviceBins {
    /// The kernel index of this width.
    fn width(&self) -> usize {
        match self {
            DeviceBins::U8(_) => 0,
            DeviceBins::U16(_) => 1,
            DeviceBins::U32(_) => 2,
        }
    }

    fn push<'a>(&'a self, launch: &mut LaunchArgs<'a>) {
        match self {
            DeviceBins::U8(b) => launch.arg(b),
            DeviceBins::U16(b) => launch.arg(b),
            DeviceBins::U32(b) => launch.arg(b),
        };
    }
}

/// The tree's gradient slice as staged on the device.
struct Staged {
    /// Address and length of the host slice staged (`len == 0`: none;
    /// `addr == 0`: gradients computed on the device, with no host copy).
    addr: usize,
    len: usize,
    grad: SumDomain,
    hess: SumDomain,
}

impl Staged {
    /// Whether the staged gradients are `gpair` (`None`: the device's own).
    fn holds(&self, gpair: Option<&[GradPair]>) -> bool {
        match gpair {
            None => self.len != 0 && self.addr == 0,
            Some(gpair) => {
                self.len != 0 && self.len == gpair.len() && self.addr == gpair.as_ptr().addr()
            }
        }
    }

    fn finite(&self) -> bool {
        self.grad.is_finite() && self.hess.is_finite()
    }

    fn sums_exact(&self, n: usize) -> bool {
        self.grad.sums_exact(n) && self.hess.sums_exact(n)
    }
}

/// Which device row list a histogram batch reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSource<'a> {
    /// The tree's partitioned rows (device-resident growth).
    Tree,
    /// Rows uploaded for one [`HistogramBackend::build`] call, with their
    /// host copy (CPU-built nodes read it).
    Upload(&'a [u32]),
}

/// Device buffers, used by one call at a time.
struct State {
    bins: DeviceBins,
    /// The same bins feature-major (`n_rows` per feature), which the
    /// partition reads one column at a time.
    cols: DeviceBins,
    /// Entries per row of `bins`.
    stride: u32,
    /// The partition's per-row directions (1 = left), by row position.
    flags: CudaSlice<u8>,
    /// The missing-value marker of `bins` (`u32::MAX` for a dense index,
    /// which no stored bin equals).
    sentinel: u32,
    /// Each feature's first global bin, then the total (`n_cols + 1`).
    feature_first: CudaSlice<u32>,
    /// Feature groups whose bins fit a block's shared memory (four `u32`s
    /// each), and the wider ones (built with global atomics).
    groups_shared: CudaSlice<u32>,
    n_shared: usize,
    groups_global: CudaSlice<u32>,
    n_global: usize,
    /// Bins of the widest shared group.
    group_bins: usize,
    /// Features of the widest group, which bounds a tile's rows.
    group_features: usize,
    /// The staged `GradPair`s, two `f32`s per row.
    gpair: CudaSlice<f32>,
    /// The staged pairs in grains, two `i64`s per row.
    units: CudaSlice<i64>,
    /// The tree's rows, partitioned in place, and the partition's scratch.
    tree_rows: CudaSlice<u32>,
    scratch: CudaSlice<u32>,
    /// Rows of the tree (`tree_rows[..tree_len]` is valid).
    tree_len: usize,
    /// Rows uploaded by a per-node build.
    upload: CudaSlice<u32>,
    /// Exact accumulators, two 64-bit words per bin per node.
    acc: CudaSlice<u64>,
    /// Per-chunk partials (integer or `f64`), two words per bin per chunk.
    partials: CudaSlice<u64>,
    /// Chunks `partials` holds.
    wave_slots: usize,
    /// The batch's histograms, two `f64`s per bin per node.
    out: CudaSlice<f64>,
    /// Descriptor uploads.
    tiles: CudaSlice<u64>,
    nodes: CudaSlice<u32>,
    totals: CudaSlice<i64>,
    segs: CudaSlice<u64>,
    rules: CudaSlice<u32>,
    table: CudaSlice<u8>,
    ptiles: CudaSlice<u64>,
    split_tiles: CudaSlice<u32>,
    tile_left: CudaSlice<u32>,
    left_len: CudaSlice<u32>,
    /// The gradient statistics the device folds.
    domain: CudaSlice<u32>,
    /// Per-block `f64` totals of a root whose blocks are not exact.
    chains: CudaSlice<f64>,
    /// Device-side rounds: the training margins, labels and weights (the
    /// latter two keyed by the host slices they were copied from), and
    /// per-leaf values.
    margins: Option<CudaSlice<f32>>,
    labels: Option<(usize, CudaSlice<f32>)>,
    weights: Option<(usize, CudaSlice<f32>)>,
    values: CudaSlice<f32>,
    staged: Staged,
    /// Page-locked staging: gradients up, histograms and rows down.
    pin_grad: Option<Pinned<f32>>,
    pin_out: Option<Pinned<f64>>,
    pin_rows: Option<Pinned<u32>>,
    /// Resident growth: the histogram slots (two `f64`s per bin per slot)
    /// and how many it holds.
    pool: CudaSlice<f64>,
    pool_slots: usize,
    /// Split-scan descriptors and results (`scan_splits` in `kernels.cu`).
    scan_tasks: CudaSlice<u32>,
    scan_totals: CudaSlice<f64>,
    scan_params: CudaSlice<f32>,
    scan_meta: CudaSlice<u32>,
    scan_acc: CudaSlice<f64>,
}

/// Page-locked host memory: copies from and to it run at full PCIe speed
/// and asynchronously (pageable copies go through a driver bounce buffer,
/// at a fraction of the bandwidth).
struct Pinned<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: plain host memory owned by this value, accessed through `&self`
// / `&mut self` borrows only.
unsafe impl<T: Send> Send for Pinned<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for Pinned<T> {}

impl<T: Copy> Pinned<T> {
    /// `len` zeroed elements (`T` must be valid all-zero, as the plain
    /// numbers used here are). `write_combined` suits buffers the host only
    /// writes and the device only reads.
    fn new(
        ctx: &Arc<CudaContext>,
        len: usize,
        write_combined: bool,
    ) -> std::result::Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let len = len.max(1);
        let bytes = len * std::mem::size_of::<T>();
        let flags = if write_combined {
            sys::CU_MEMHOSTALLOC_WRITECOMBINED
        } else {
            0
        };
        // SAFETY: allocates `bytes` bytes of page-locked host memory, owned
        // by the returned value and freed on drop.
        let raw = unsafe { cudarc::driver::result::malloc_host(bytes, flags) }?;
        let ptr = std::ptr::NonNull::new(raw.cast::<T>())
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        // SAFETY: the allocation holds `bytes` writable bytes, suitably
        // aligned for `T` (page-aligned).
        unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, bytes) };
        Ok(Pinned {
            ptr,
            len,
            ctx: ctx.clone(),
        })
    }

    fn as_slice(&self) -> &[T] {
        // SAFETY: `len` initialized elements owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `len` initialized elements owned exclusively by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for Pinned<T> {
    fn drop(&mut self) {
        // Every copy touching the buffer was synchronized before its call
        // returned, so nothing reads or writes it any more.
        let _ = self.ctx.bind_to_thread();
        // SAFETY: allocated by `malloc_host`, freed once.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.as_ptr().cast()) };
    }
}

/// A pinned buffer of at least `len` elements in `slot`.
fn pinned<'a, T: Copy>(
    stream: &Arc<CudaStream>,
    slot: &'a mut Option<Pinned<T>>,
    len: usize,
    write_combined: bool,
) -> std::result::Result<&'a mut Pinned<T>, DriverError> {
    if slot.as_ref().is_none_or(|p| p.len < len) {
        *slot = None;
        *slot = Some(Pinned::new(
            stream.context(),
            len.next_power_of_two(),
            write_combined,
        )?);
    }
    Ok(slot.as_mut().expect("just allocated"))
}

/// Copy `src` into `dst`'s front through the pinned buffer `staging`, in
/// pieces: the host fills piece `k + 1` (in parallel) while the DMA engine
/// moves piece `k`. Asynchronous: the caller synchronizes before `staging`
/// is reused.
fn upload_pinned<T: Copy + Send + Sync + DeviceRepr>(
    stream: &Arc<CudaStream>,
    staging: &mut Pinned<T>,
    src: &[T],
    dst: &mut CudaSlice<T>,
) -> std::result::Result<(), DriverError> {
    /// Bytes per DMA piece.
    const PIECE_BYTES: usize = 8 << 20;
    assert!(src.len() <= staging.len && src.len() <= dst.len());
    let piece = (PIECE_BYTES / std::mem::size_of::<T>()).max(1);
    let (base, _record) = dst.device_ptr_mut(stream);
    let host = &mut staging.as_mut_slice()[..src.len()];
    for (k, (h, s)) in host.chunks_mut(piece).zip(src.chunks(piece)).enumerate() {
        h.par_chunks_mut(1 << 16)
            .zip(s.par_chunks(1 << 16))
            .for_each(|(a, b)| a.copy_from_slice(b));
        let offset = (k * piece * std::mem::size_of::<T>()) as u64;
        // SAFETY: `h` is pinned host memory that stays untouched until the
        // caller synchronizes; the destination range lies within `dst`.
        unsafe { cudarc::driver::result::memcpy_htod_async(base + offset, h, stream.cu_stream()) }?;
    }
    Ok(())
}

/// Copy `len` elements from `src`'s front into `staging` and wait for them.
fn download_pinned<'a, T: Copy + DeviceRepr>(
    stream: &Arc<CudaStream>,
    staging: &'a mut Pinned<T>,
    src: &CudaSlice<T>,
    len: usize,
) -> std::result::Result<&'a [T], DriverError> {
    assert!(len <= staging.len && len <= src.len());
    if len > 0 {
        let (ptr, _record) = src.device_ptr(stream);
        let host = &mut staging.as_mut_slice()[..len];
        // SAFETY: pinned destination of `len` elements, read only after
        // the synchronization below.
        unsafe { cudarc::driver::result::memcpy_dtoh_async(host, ptr, stream.cu_stream()) }?;
    }
    stream.synchronize()?;
    Ok(&staging.as_slice()[..len])
}

/// The CUDA histogram backend: implements [`HistogramBackend`] on an NVIDIA
/// GPU, and grows hist trees with their rows on the GPU (see the
/// [module docs](self)). Constructed once per training run (the index
/// upload is per-dataset); the gradient slice is uploaded by
/// [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = cuda`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`]. Its
/// histograms equal the CPU backend's bit for bit. `build` must receive the
/// index the backend was built from, and the gradient slice must not change
/// between `prepare` and the tree's last `build`; inputs that do not fit
/// the backend's buffers (another index shape, a gradient slice of another
/// length, row indices past the index) never reach the GPU and take the
/// CPU path, which checks them.
pub struct CudaHistBackend {
    device: Arc<Device>,
    n_rows: usize,
    n_cols: usize,
    total_bins: usize,
    /// Whether the index is dense (no missing values).
    dense: bool,
    state: Mutex<State>,
    counters: Counters,
}

impl std::fmt::Debug for CudaHistBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHistBackend")
            .field("device", &self.device.name)
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .field("total_bins", &self.total_bins)
            .finish_non_exhaustive()
    }
}

/// A device buffer of at least `len` elements: `buf` itself, or a fresh
/// (zeroed) one replacing it, sized up to a power of two so repeated growth
/// stays rare.
fn fit<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<T>,
    len: usize,
) -> std::result::Result<(), DriverError> {
    if buf.len() < len {
        *buf = stream.alloc_zeros(len.next_power_of_two())?;
    }
    Ok(())
}

/// Copy `data` to the front of `buf`, growing it as needed.
fn upload<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<T>,
    data: &[T],
) -> std::result::Result<(), DriverError> {
    if data.is_empty() {
        return Ok(());
    }
    fit(stream, buf, data.len())?;
    stream.memcpy_htod(data, &mut buf.slice_mut(..data.len()))
}

/// Feature groups over `bins_of` (bins per feature): contiguous features
/// whose bins fit `cap`, balanced, and the features wider than `cap` alone.
/// Each group is `[first feature, end feature, first global bin, bins]`.
fn feature_groups(bins_of: &[usize], cap: usize) -> (Vec<u32>, Vec<u32>) {
    let total: usize = bins_of.iter().filter(|&&b| b <= cap).sum();
    let target = total.div_ceil(total.div_ceil(cap.max(1)).max(1)).max(1);
    let (mut shared, mut global) = (Vec::new(), Vec::new());
    let mut first_bin = 0usize;
    let mut start: Option<(usize, usize)> = None;
    let mut open_bins = 0usize;
    let close = |out: &mut Vec<u32>, (f0, b0): (usize, usize), f1: usize, bins: usize| {
        out.extend([f0 as u32, f1 as u32, b0 as u32, bins as u32]);
    };
    for (f, &bins) in bins_of.iter().enumerate() {
        if bins > cap {
            if let Some(open) = start.take() {
                close(&mut shared, open, f, open_bins);
            }
            close(&mut global, (f, first_bin), f + 1, bins);
        } else {
            if let Some(open) = start
                && (open_bins + bins > cap || open_bins >= target)
            {
                close(&mut shared, open, f, open_bins);
                start = None;
            }
            if start.is_none() {
                start = Some((f, first_bin));
                open_bins = 0;
            }
            open_bins += bins;
        }
        first_bin += bins;
    }
    if let Some(open) = start {
        close(&mut shared, open, bins_of.len(), open_bins);
    }
    (shared, global)
}

/// The feature-local bins of `index`, with `missing` where a row lacks a
/// feature: row-major with `stride` entries per row (the histogram
/// kernels' layout; padding unread), and feature-major (`n_rows` per
/// feature, the partition's layout).
fn local_bins<T: Copy + Send + Sync>(
    index: &GHistIndex,
    missing: T,
    stride: usize,
    narrow: impl Fn(u32) -> T + Sync,
) -> (Vec<T>, Vec<T>) {
    let n_cols = index.n_cols();
    let n_rows = index.n_rows();
    let cuts = index.cuts();
    let first: Vec<u32> = (0..n_cols).map(|f| cuts.feature_bins(f).0 as u32).collect();
    let mut feature_of = vec![0u32; index.total_bins()];
    for f in 0..n_cols {
        let (fs, fe) = cuts.feature_bins(f);
        feature_of[fs..fe].fill(f as u32);
    }
    let row_ptr = index.row_ptr();
    let mut rows = vec![missing; n_rows * stride];
    let fill = |row: &mut [T], global: u32| {
        let f = feature_of[global as usize] as usize;
        row[f] = narrow(global - first[f]);
    };
    match index.bins() {
        Bins::U16(bins) => rows
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(r, row)| {
                for &b in &bins[row_ptr[r]..row_ptr[r + 1]] {
                    fill(row, u32::from(b));
                }
            }),
        Bins::U32(bins) => rows
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(r, row)| {
                for &b in &bins[row_ptr[r]..row_ptr[r + 1]] {
                    fill(row, b);
                }
            }),
    }
    let mut cols = vec![missing; n_rows * n_cols];
    cols.par_chunks_mut(n_rows)
        .enumerate()
        .for_each(|(f, col)| {
            for (r, c) in col.iter_mut().enumerate() {
                *c = rows[r * stride + f];
            }
        });
    (rows, cols)
}

/// Row stride (entries) of `n_cols` bins of `width` bytes: rows start on
/// 32-byte boundaries, so a row's bins span the fewest memory sectors.
fn row_stride(n_cols: usize, width: usize) -> usize {
    (n_cols * width).div_ceil(32) * 32 / width
}

impl CudaHistBackend {
    /// Build the backend for `index` on CUDA device `ordinal`: upload its
    /// bins and allocate the per-tree buffers.
    pub fn new(index: &GHistIndex, ordinal: usize) -> Result<Self> {
        let device = device(ordinal).map_err(HessboostError::gpu)?;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        if total_bins == 0 || n_rows == 0 || n_cols == 0 {
            return Err(HessboostError::invalid_data(
                "data",
                "the CUDA backend needs a non-empty binned dataset",
            ));
        }
        if u32::try_from(n_rows).is_err() || u32::try_from(total_bins).is_err() {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the CUDA backend indexes rows and bins in 32 bits \
                     ({n_rows} rows, {total_bins} bins)"
                ),
            ));
        }
        let state = Self::upload(&device, index).map_err(gpu_error)?;
        Ok(CudaHistBackend {
            n_rows,
            n_cols,
            total_bins,
            dense: index.dense_stride().is_some(),
            device,
            state: Mutex::new(state),
            counters: Counters::default(),
        })
    }

    /// The name of the device the backend runs on.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device.name
    }

    /// The nodes (and rows) built so far with each strategy.
    #[must_use]
    pub fn node_counts(&self) -> NodeCounts {
        self.counters.snapshot()
    }

    fn upload(device: &Device, index: &GHistIndex) -> std::result::Result<State, DriverError> {
        let stream = &device.stream;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        let cuts = index.cuts();
        let bins_of: Vec<usize> = (0..n_cols)
            .map(|f| {
                let (fs, fe) = cuts.feature_bins(f);
                fe - fs
            })
            .collect();
        let widest = bins_of.iter().copied().max().unwrap_or(0);
        let dense = index.dense_stride().is_some();
        // The narrowest width whose values cover every feature's bins and,
        // with missing values, a sentinel above them.
        let room = |max: usize| {
            if dense {
                widest <= max + 1
            } else {
                widest <= max
            }
        };
        let (bins, cols, sentinel, stride) = if room(u8::MAX as usize) {
            let missing = u8::MAX;
            let stride = row_stride(n_cols, 1);
            let (rows, cols) = local_bins(index, missing, stride, |b| b as u8);
            (
                DeviceBins::U8(stream.clone_htod(&rows)?),
                DeviceBins::U8(stream.clone_htod(&cols)?),
                u32::from(missing),
                stride,
            )
        } else if room(u16::MAX as usize) {
            let missing = u16::MAX;
            let stride = row_stride(n_cols, 2);
            let (rows, cols) = local_bins(index, missing, stride, |b| b as u16);
            (
                DeviceBins::U16(stream.clone_htod(&rows)?),
                DeviceBins::U16(stream.clone_htod(&cols)?),
                u32::from(missing),
                stride,
            )
        } else {
            let missing = u32::MAX;
            let stride = row_stride(n_cols, 4);
            let (rows, cols) = local_bins(index, missing, stride, |b| b);
            (
                DeviceBins::U32(stream.clone_htod(&rows)?),
                DeviceBins::U32(stream.clone_htod(&cols)?),
                missing,
                stride,
            )
        };
        let sentinel = if dense { u32::MAX } else { sentinel };
        let mut first: Vec<u32> = (0..n_cols).map(|f| cuts.feature_bins(f).0 as u32).collect();
        first.push(total_bins as u32);
        let cap = device.shared_bytes / 16;
        let (shared, global) = feature_groups(&bins_of, cap);
        let group_features = shared
            .chunks(4)
            .chain(global.chunks(4))
            .map(|g| (g[1] - g[0]) as usize)
            .max()
            .unwrap_or(1);
        let group_bins = shared.chunks(4).map(|g| g[3] as usize).max().unwrap_or(0);
        let alloc_u32 = |len: usize| stream.alloc_zeros::<u32>(len.max(1));
        let alloc_u64 = |len: usize| stream.alloc_zeros::<u64>(len.max(1));
        let mut groups_shared = alloc_u32(shared.len())?;
        upload(stream, &mut groups_shared, &shared)?;
        let mut groups_global = alloc_u32(global.len())?;
        upload(stream, &mut groups_global, &global)?;
        Ok(State {
            bins,
            cols,
            stride: stride as u32,
            flags: stream.alloc_zeros(n_rows)?,
            sentinel,
            feature_first: stream.clone_htod(&first)?,
            n_shared: shared.len() / 4,
            groups_shared,
            n_global: global.len() / 4,
            groups_global,
            group_bins,
            group_features,
            gpair: stream.alloc_zeros(n_rows * 2)?,
            units: stream.alloc_zeros(n_rows * 2)?,
            tree_rows: stream.alloc_zeros(n_rows)?,
            scratch: stream.alloc_zeros(n_rows)?,
            tree_len: 0,
            upload: stream.alloc_zeros(n_rows)?,
            acc: alloc_u64(total_bins * 2)?,
            partials: alloc_u64(1)?,
            wave_slots: (PARTIAL_BYTES / (total_bins * 16)).max(1),
            out: stream.alloc_zeros(total_bins * 2)?,
            tiles: alloc_u64(1)?,
            nodes: alloc_u32(1)?,
            totals: stream.alloc_zeros(2)?,
            segs: alloc_u64(1)?,
            rules: alloc_u32(1)?,
            table: stream.alloc_zeros(1)?,
            ptiles: alloc_u64(1)?,
            split_tiles: alloc_u32(1)?,
            tile_left: alloc_u32(1)?,
            left_len: alloc_u32(1)?,
            domain: alloc_u32(6)?,
            chains: stream.alloc_zeros(2)?,
            margins: None,
            labels: None,
            weights: None,
            values: stream.alloc_zeros(1)?,
            staged: Staged {
                addr: 0,
                len: 0,
                grad: SumDomain::EMPTY,
                hess: SumDomain::EMPTY,
            },
            pin_grad: None,
            pin_out: None,
            pin_rows: None,
            pool: stream.alloc_zeros(2)?,
            pool_slots: 0,
            scan_tasks: alloc_u32(4)?,
            scan_totals: stream.alloc_zeros(2)?,
            scan_params: stream.alloc_zeros(3)?,
            scan_meta: alloc_u32(4)?,
            scan_acc: stream.alloc_zeros(2)?,
        })
    }

    /// Whether `ghist` is the index the backend was built from (by shape).
    fn fits(&self, ghist: &GHistIndex) -> bool {
        ghist.n_rows() == self.n_rows
            && ghist.n_cols() == self.n_cols
            && ghist.total_bins() == self.total_bins
            && ghist.dense_stride().is_some() == self.dense
    }

    /// Lock the device state, unless the device has failed.
    fn lock(&self) -> Option<std::sync::MutexGuard<'_, State>> {
        if self.device.failed.load(Ordering::Acquire) {
            return None;
        }
        Some(
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// `result`'s value, or `None` after marking the device failed (CUDA
    /// errors are sticky).
    fn ok<T>(&self, result: std::result::Result<T, DriverError>) -> Option<T> {
        result
            .inspect_err(|_| self.device.failed.store(true, Ordering::Release))
            .ok()
    }

    /// Stage `gpair` on the device with its exactness statistics. A slice
    /// of any length other than `n_rows` is not staged.
    fn stage(&self, state: &mut State, gpair: &[GradPair]) -> std::result::Result<(), DriverError> {
        // Unstage first, so a slice that fails to upload is never mistaken
        // for the previous one.
        state.staged.len = 0;
        if gpair.len() != self.n_rows {
            return Ok(());
        }
        let grad = SumDomain::of_slice(gpair, |p| p.grad);
        let hess = SumDomain::of_slice(gpair, |p| p.hess);
        // SAFETY: `GradPair` is `repr(C)` of two `f32`s, so the slice is
        // `2 * len` contiguous `f32`s.
        let flat =
            unsafe { std::slice::from_raw_parts(gpair.as_ptr().cast::<f32>(), gpair.len() * 2) };
        let stream = &self.device.stream;
        let staging = pinned(stream, &mut state.pin_grad, flat.len(), true)?;
        upload_pinned(stream, staging, flat, &mut state.gpair)?;
        self.stage_units(state, grad, hess)?;
        state.staged.addr = gpair.as_ptr().addr();
        Ok(())
    }

    /// Convert the device's gradient pairs into grains of `grad` and
    /// `hess`, the statistics of them, and record them as staged (from the
    /// device, until the caller sets a host address).
    fn stage_units(
        &self,
        state: &mut State,
        grad: SumDomain,
        hess: SumDomain,
    ) -> std::result::Result<(), DriverError> {
        let stream = &self.device.stream;
        let n = self.n_rows as u64;
        let (to_grad, to_hess) = (grad.unit_scale(), hess.unit_scale());
        let mut launch = stream.launch_builder(&self.device.kernels.stage_units);
        launch
            .arg(&state.gpair)
            .arg(&mut state.units)
            .arg(&n)
            .arg(&to_grad)
            .arg(&to_hess);
        // SAFETY: the kernel reads `n` `float2`s of `gpair` and writes `n`
        // `longlong2`s of `units`, both sized `2 * n_rows` words, and takes
        // `(u64, f64, f64)` scalars as passed.
        unsafe { launch.launch(self.device.grid(self.n_rows)) }?;
        state.staged = Staged {
            addr: 0,
            len: self.n_rows,
            grad,
            hess,
        };
        Ok(())
    }

    /// The statistics of the device's gradient pairs, folded on the device.
    fn device_domains(
        &self,
        state: &mut State,
    ) -> std::result::Result<(SumDomain, SumDomain), DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        upload(stream, &mut state.domain, &[0, u32::MAX, 1, 0, u32::MAX, 1])?;
        let n = self.n_rows as u64;
        let mut launch = stream.launch_builder(&device.kernels.grad_domain);
        launch.arg(&state.gpair).arg(&n).arg(&mut state.domain);
        // SAFETY: reads `n_rows` pairs and folds into six words.
        unsafe { launch.launch(device.grid(self.n_rows)) }?;
        let mut host = [0u32; 6];
        stream.memcpy_dtoh(&state.domain.slice(..6), &mut host[..])?;
        stream.synchronize()?;
        Ok((
            SumDomain::from_device(host[0], host[1], host[2] != 0),
            SumDomain::from_device(host[3], host[4], host[5] != 0),
        ))
    }

    /// Launch the integer histogram kernels over `n_tiles` tiles (uploaded
    /// to `state.tiles`) reading `source`.
    fn launch_tiles(
        &self,
        state: &mut State,
        source: RowSource<'_>,
        n_tiles: usize,
    ) -> std::result::Result<(), DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        let State {
            bins,
            stride,
            sentinel,
            feature_first,
            groups_shared,
            n_shared,
            groups_global,
            n_global,
            group_bins,
            units,
            tree_rows,
            upload,
            acc,
            partials,
            tiles,
            ..
        } = state;
        let rows = match source {
            RowSource::Tree => &*tree_rows,
            RowSource::Upload(_) => &*upload,
        };
        let stride = *stride;
        let total_bins = self.total_bins as u64;
        let w = bins.width();
        for (kernel, groups, n_groups, shared) in [
            (
                &device.kernels.hist_shared[w],
                &*groups_shared,
                *n_shared,
                *group_bins * 16,
            ),
            (
                &device.kernels.hist_global[w],
                &*groups_global,
                *n_global,
                0,
            ),
        ] {
            if n_groups == 0 {
                continue;
            }
            let groups_u32 = n_groups as u32;
            let mut launch = stream.launch_builder(kernel);
            bins.push(&mut launch);
            launch
                .arg(&stride)
                .arg(&*sentinel)
                .arg(&*feature_first)
                .arg(rows)
                .arg(&*tiles)
                .arg(groups)
                .arg(&*units)
                .arg(&mut *acc)
                .arg(&mut *partials)
                .arg(&total_bins)
                .arg(&groups_u32);
            let config = LaunchConfig {
                grid_dim: ((n_tiles * n_groups) as u32, 1, 1),
                block_dim: (HIST_THREADS, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            // SAFETY: every tile lists rows of `rows` below `n_rows` (the
            // callers' segments lie in the valid prefix), each group's
            // shared histogram fits the dynamic allocation, tile targets
            // index slots `acc` and `partials` were sized for, and every
            // stored local bin plus its feature's first bin is below
            // `total_bins`. Arguments match the kernel's parameters.
            unsafe { launch.launch(config) }?;
        }
        Ok(())
    }

    /// The histograms of `nodes` (`(rows of source, contiguous)`), reading
    /// rows from `source` (an upload's host copy feeds CPU-built nodes).
    /// With `targets`, node
    /// `k`'s histogram is written to slot `targets[k]` of `state.out`
    /// (which the caller has made the resident pool) and nothing is read
    /// back (`Some` of an empty list).
    fn histograms_on(
        &self,
        state: &mut State,
        source: RowSource<'_>,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[Segment],
        targets: Option<&[HistSlot]>,
    ) -> std::result::Result<Option<Vec<Histogram>>, DriverError> {
        let bins = self.total_bins;
        let device = &*self.device;
        let stream = &device.stream;
        let mut results: Vec<Option<Histogram>> = (0..nodes.len()).map(|_| None).collect();
        let mut exact = Vec::new();
        let mut chunked = Vec::new();
        let mut chains = Vec::new();
        let mut cpu = Vec::new();
        let mut slots = Vec::new();
        // Output slot of node `k`, the next free one without `targets`.
        let slot_of = |k: usize, next: usize| targets.map_or(next, |t| t[k] as usize);
        // Rows per tile must keep `rows * features` of a group in a `u32`.
        // (Halved: the kernel's unrolled indices run up to four block widths
        // past the tile's last element.)
        let max_tile = (u32::MAX as usize / 2 / state.group_features.max(1)).max(1);
        for (k, &seg) in nodes.iter().enumerate() {
            if seg.len == 0 {
                match targets {
                    Some(t) => {
                        let s = t[k] as usize;
                        stream.memset_zeros(
                            &mut state.out.slice_mut(s * bins * 2..(s + 1) * bins * 2),
                        )?;
                    }
                    None => results[k] = Some(zeroed(bins)),
                }
                self.counters.count(Strategy::Exact, 0);
                continue;
            }
            let order = sum_order(seg.len);
            let mut strategy = plan(&state.staged, order, seg.len);
            if let (Strategy::ExactChunks | Strategy::Chains, SumOrder::Blocked { grain }) =
                (strategy, order)
                && grain > max_tile
            {
                strategy = Strategy::Cpu;
            }
            self.counters.count(strategy, seg.len);
            let slot = slot_of(k, slots.len());
            match (strategy, order) {
                (Strategy::Exact, _) => exact.push((k, seg, slot)),
                (Strategy::ExactChunks, SumOrder::Blocked { grain }) => {
                    chunked.push((k, seg, grain, slot));
                }
                (Strategy::Chains, SumOrder::Blocked { grain }) => {
                    chains.push((k, seg, grain, slot));
                }
                (Strategy::Chains, SumOrder::Chain) => chains.push((k, seg, seg.len, slot)),
                _ => {
                    cpu.push(k);
                    continue;
                }
            }
            slots.push(k);
        }
        if !slots.is_empty() && targets.is_none() {
            fit(stream, &mut state.out, slots.len() * bins * 2)?;
        }
        let (grad_value, hess_value) = (
            state.staged.grad.value_scale(),
            state.staged.hess.value_scale(),
        );

        // Strategy 1: exact integers, any tiling, into per-node accumulators.
        for batch in exact.chunks(MAX_GRID_Y) {
            fit(stream, &mut state.acc, batch.len() * bins * 2)?;
            stream.memset_zeros(&mut state.acc.slice_mut(..batch.len() * bins * 2))?;
            let tile_rows = HIST_TILE.min(max_tile);
            let mut tiles = Vec::new();
            for (j, &(_, seg, _)) in batch.iter().enumerate() {
                for start in (0..seg.len).step_by(tile_rows) {
                    let count = tile_rows.min(seg.len - start);
                    tiles.extend([(seg.offset + start) as u64, count as u64 | (j as u64) << 32]);
                }
            }
            upload(stream, &mut state.tiles, &tiles)?;
            self.launch_tiles(state, source, tiles.len() / 2)?;
            let pairs: Vec<u32> = batch
                .iter()
                .enumerate()
                .flat_map(|(j, &(_, _, slot))| [j as u32, slot as u32])
                .collect();
            upload(stream, &mut state.nodes, &pairs)?;
            let total_bins = bins as u64;
            let mut launch = stream.launch_builder(&device.kernels.finalize_exact);
            launch
                .arg(&state.acc)
                .arg(&state.nodes)
                .arg(&total_bins)
                .arg(&grad_value)
                .arg(&hess_value)
                .arg(&mut state.out);
            // SAFETY: reads `batch.len()` accumulators and writes their
            // output slots, both within the sized buffers.
            unsafe { launch.launch(Device::per_node_bins(bins, batch.len())) }?;
        }

        // Strategy 2: exact chunks, the CPU's chunks, reduced in order.
        let chunk_list: Vec<(usize, usize, usize, usize)> = chunked
            .iter()
            .flat_map(|&(_, seg, grain, slot)| {
                (0..seg.len.div_ceil(grain)).map(move |c| {
                    let start = c * grain;
                    (seg.offset + start, grain.min(seg.len - start), slot, c)
                })
            })
            .collect();
        if !chunk_list.is_empty() {
            fit(
                stream,
                &mut state.partials,
                state.wave_slots.min(chunk_list.len()) * bins * 2,
            )?;
        }
        for wave in chunk_list.chunks(state.wave_slots) {
            let tiles: Vec<u64> = wave
                .iter()
                .enumerate()
                .flat_map(|(s, &(begin, count, _, _))| {
                    [begin as u64, count as u64 | ((s as u64) | 1 << 31) << 32]
                })
                .collect();
            upload(stream, &mut state.tiles, &tiles)?;
            self.launch_tiles(state, source, wave.len())?;
            // Consecutive chunks of one node reduce together.
            let mut ranges: Vec<u32> = Vec::new();
            let mut s = 0;
            while s < wave.len() {
                let (_, _, slot, c) = wave[s];
                let mut e = s + 1;
                while e < wave.len() && wave[e].2 == slot {
                    e += 1;
                }
                ranges.extend([s as u32, (e - s) as u32, slot as u32, u32::from(c == 0)]);
                s = e;
            }
            upload(stream, &mut state.nodes, &ranges)?;
            let total_bins = bins as u64;
            for batch in 0..(ranges.len() / 4).div_ceil(MAX_GRID_Y) {
                let first = batch * MAX_GRID_Y;
                let count = (ranges.len() / 4 - first).min(MAX_GRID_Y);
                let view = state.nodes.slice(first * 4..(first + count) * 4);
                let mut launch = stream.launch_builder(&device.kernels.reduce_chunks);
                launch
                    .arg(&state.partials)
                    .arg(&view)
                    .arg(&total_bins)
                    .arg(&grad_value)
                    .arg(&hess_value)
                    .arg(&mut state.out);
                // SAFETY: each range reads partial slots of this wave and
                // writes its node's output slot.
                unsafe { launch.launch(Device::per_node_bins(bins, count)) }?;
            }
        }

        // Strategy 3: `f64` chains, one thread per (chunk, feature).
        for &(_, seg, seg_rows, slot) in &chains {
            let segs = seg.len.div_ceil(seg_rows);
            let wave_chunks = state.wave_slots;
            fit(
                stream,
                &mut state.partials,
                wave_chunks.min(segs) * bins * 2,
            )?;
            for (w, first) in (0..segs).step_by(wave_chunks).enumerate() {
                let wave = wave_chunks.min(segs - first);
                let begin = seg.offset + first * seg_rows;
                let end = (seg.offset + (first + wave) * seg_rows).min(seg.offset + seg.len);
                stream.memset_zeros(&mut state.partials.slice_mut(..wave * bins * 2))?;
                let State {
                    bins: dev_bins,
                    stride,
                    sentinel,
                    feature_first,
                    gpair: dev_gpair,
                    tree_rows,
                    upload: uploaded,
                    partials,
                    out,
                    ..
                } = &mut *state;
                let rows = match source {
                    RowSource::Tree => tree_rows.slice(begin..end),
                    RowSource::Upload(_) => uploaded.slice(begin..end),
                };
                let (n, seg_rows64, segs64) = ((end - begin) as u64, seg_rows as u64, wave as u64);
                let n_cols = self.n_cols as u32;
                let total_bins = bins as u64;
                let mut launch =
                    stream.launch_builder(&device.kernels.hist_chain[dev_bins.width()]);
                dev_bins.push(&mut launch);
                launch
                    .arg(&*stride)
                    .arg(&n_cols)
                    .arg(&*sentinel)
                    .arg(&*feature_first)
                    .arg(&rows)
                    .arg(&n)
                    .arg(&seg_rows64)
                    .arg(&segs64)
                    .arg(&*dev_gpair)
                    .arg(&mut *partials)
                    .arg(&total_bins);
                // SAFETY: one thread per (chunk, feature) of the wave over
                // listed rows below `n_rows`; each chunk's partial fits.
                unsafe { launch.launch(Device::one_per(wave * self.n_cols)) }?;
                let init = i32::from(w == 0);
                let mut target = out.slice_mut(slot * bins * 2..(slot + 1) * bins * 2);
                let mut reduce = stream.launch_builder(&device.kernels.reduce_chains);
                reduce
                    .arg(&*partials)
                    .arg(&segs64)
                    .arg(&total_bins)
                    .arg(&init)
                    .arg(&mut target);
                // SAFETY: reads `wave` partials and writes one output slot.
                unsafe { reduce.launch(device.grid(bins)) }?;
            }
        }

        // Strategy 4 on the CPU, while the GPU works: needs the host
        // gradients (callers without them checked there are no such nodes).
        for &k in &cpu {
            let Some(gpair) = gpair else {
                return Ok(None);
            };
            let seg = nodes[k];
            let rows = if let RowSource::Upload(rows) = source {
                rows[seg.offset..seg.offset + seg.len].to_vec()
            } else {
                let mut rows = vec![0u32; seg.len];
                stream.memcpy_dtoh(
                    &state.tree_rows.slice(seg.offset..seg.offset + seg.len),
                    &mut rows,
                )?;
                rows
            };
            let mut hist = zeroed(bins);
            CpuBackend.build(ghist, &rows, gpair, &mut hist);
            match targets {
                Some(t) => {
                    let s = t[k] as usize;
                    let flat: Vec<f64> = hist.iter().flat_map(|b| [b.grad, b.hess]).collect();
                    stream.memcpy_htod(
                        &flat,
                        &mut state.out.slice_mut(s * bins * 2..(s + 1) * bins * 2),
                    )?;
                }
                None => results[k] = Some(hist),
            }
        }
        if targets.is_some() {
            return Ok(Some(Vec::new()));
        }

        if !slots.is_empty() {
            let staging = pinned(stream, &mut state.pin_out, slots.len() * bins * 2, false)?;
            // Kernel faults surface at this synchronization point.
            let all = download_pinned(stream, staging, &state.out, slots.len() * bins * 2)?;
            // SAFETY: `GradStats` is `repr(C)` of two `f64`s, so `all` is
            // `slots.len() * bins` contiguous `GradStats`.
            let all = unsafe {
                std::slice::from_raw_parts(all.as_ptr().cast::<GradStats>(), slots.len() * bins)
            };
            // Fresh histograms fault their pages in on first write: copy
            // them out in parallel.
            let copied: Vec<Histogram> = (0..slots.len())
                .into_par_iter()
                .map(|slot| all[slot * bins..(slot + 1) * bins].to_vec())
                .collect();
            for (hist, &k) in copied.into_iter().zip(&slots) {
                results[k] = Some(hist);
            }
        }
        Ok(Some(
            results.into_iter().map(Option::unwrap_or_default).collect(),
        ))
    }

    fn partition_on(
        &self,
        state: &mut State,
        splits: &[RowSplit<'_>],
    ) -> std::result::Result<Vec<Partitioned>, DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        let mut segs = Vec::with_capacity(splits.len() * 2);
        let mut rules = Vec::with_capacity(splits.len() * 4);
        let mut table: Vec<u8> = Vec::new();
        let mut ptiles = Vec::new();
        let mut split_tiles = Vec::with_capacity(splits.len() * 2);
        for (s, split) in splits.iter().enumerate() {
            segs.extend([split.seg.offset as u64, split.seg.len as u64]);
            let (limit, table_at, kind) = match split.rule {
                RowRule::Below(limit) => (limit, 0, 0),
                RowRule::Table(left) => {
                    let at = table.len() as u32;
                    table.extend(left.iter().map(|&l| u8::from(l)));
                    (0, at, 2)
                }
            };
            rules.extend([
                split.feature,
                limit,
                table_at,
                kind | u32::from(split.default_left),
            ]);
            let n = split.seg.len.div_ceil(PART_TILE);
            split_tiles.extend([ptiles.len() as u32, n as u32]);
            ptiles.extend((0..n).map(|k| (s as u64) << 32 | k as u64));
        }
        let n_splits = splits.len() as u32;
        let n_tiles = ptiles.len();
        upload(stream, &mut state.segs, &segs)?;
        upload(stream, &mut state.rules, &rules)?;
        upload(stream, &mut state.table, &table)?;
        upload(stream, &mut state.split_tiles, &split_tiles)?;
        fit(stream, &mut state.left_len, splits.len())?;
        let State {
            cols,
            flags,
            sentinel,
            tree_rows,
            scratch,
            segs: d_segs,
            rules: d_rules,
            table: d_table,
            ptiles: d_ptiles,
            split_tiles: d_split_tiles,
            tile_left,
            left_len,
            ..
        } = state;
        let n_rows = self.n_rows as u64;
        let w = cols.width();
        if n_tiles > 0 {
            upload(stream, d_ptiles, &ptiles)?;
            fit(stream, tile_left, n_tiles)?;
            let tiles_config = LaunchConfig {
                grid_dim: (n_tiles as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut count = stream.launch_builder(&device.kernels.route_count[w]);
            cols.push(&mut count);
            count
                .arg(&n_rows)
                .arg(&*sentinel)
                .arg(&*d_segs)
                .arg(&*d_rules)
                .arg(&*d_table)
                .arg(&*d_ptiles)
                .arg(&*tree_rows)
                .arg(&mut *flags)
                .arg(&mut *tile_left);
            // SAFETY: one block per tile; every segment lies in the tree's
            // valid rows, every row id is below `n_rows`, and each rule's
            // table range covers its feature's bins.
            unsafe { count.launch(tiles_config) }?;
        }
        let mut scan = stream.launch_builder(&device.kernels.route_scan);
        scan.arg(&*d_split_tiles)
            .arg(&n_splits)
            .arg(&mut *tile_left)
            .arg(&mut *left_len);
        // SAFETY: one thread per split over its own tiles' counts.
        unsafe { scan.launch(Device::one_per(splits.len())) }?;
        if n_tiles > 0 {
            let tiles_config = LaunchConfig {
                grid_dim: (n_tiles as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut scatter = stream.launch_builder(&device.kernels.route_scatter);
            scatter
                .arg(&*d_segs)
                .arg(&*d_ptiles)
                .arg(&*tree_rows)
                .arg(&*flags)
                .arg(&*tile_left)
                .arg(&*left_len)
                .arg(&mut *scratch);
            // SAFETY: as for the count; each row lands inside its own
            // segment of `scratch` (`n_rows` long).
            unsafe { scatter.launch(tiles_config) }?;
            let mut copy = stream.launch_builder(&device.kernels.route_copy);
            copy.arg(&*d_segs)
                .arg(&*d_ptiles)
                .arg(&*scratch)
                .arg(&mut *tree_rows);
            // SAFETY: copies each tile's span within its segment.
            unsafe { copy.launch(tiles_config) }?;
        }
        let mut host = vec![0u32; splits.len()];
        stream.memcpy_dtoh(&left_len.slice(..host.len()), &mut host)?;
        stream.synchronize()?;
        Ok(splits
            .iter()
            .zip(host)
            .map(|(split, left_len)| {
                let left_len = left_len as usize;
                Partitioned {
                    left: Segment {
                        offset: split.seg.offset,
                        len: left_len,
                    },
                    right: Segment {
                        offset: split.seg.offset + left_len,
                        len: split.seg.len - left_len,
                    },
                }
            })
            .collect())
    }
}

/// The strategy for a node of `n` rows summed in `order` (the
/// [module docs](self)' numbering).
fn plan(staged: &Staged, order: SumOrder, n: usize) -> Strategy {
    // NaN payloads are not portable between the CPU's and the GPU's
    // arithmetic, so a non-finite slice keeps the CPU's bits by running there.
    if !staged.finite() {
        return Strategy::Cpu;
    }
    if staged.sums_exact(n) {
        return Strategy::Exact;
    }
    match order {
        SumOrder::Blocked { grain } if staged.sums_exact(grain) => Strategy::ExactChunks,
        SumOrder::Blocked { .. } | SumOrder::Chain => Strategy::Chains,
    }
}

fn gpu_error(error: DriverError) -> HessboostError {
    HessboostError::gpu(format!("CUDA: {error}"))
}

impl HistogramBackend for CudaHistBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let fits = self.fits(ghist)
            && out.len() == self.total_bins
            && rows.len() <= self.n_rows
            && rows.iter().all(|&r| (r as usize) < self.n_rows);
        let built = fits.then(|| self.lock()).flatten().and_then(|mut state| {
            let state = &mut *state;
            if !state.staged.holds(Some(gpair)) {
                self.ok(self.stage(state, gpair))?;
                if !state.staged.holds(Some(gpair)) {
                    return None;
                }
            }
            let stream = &self.device.stream;
            self.ok(upload(stream, &mut state.upload, rows))?;
            let node = Segment {
                offset: 0,
                len: rows.len(),
            };
            self.ok(self.histograms_on(
                state,
                RowSource::Upload(rows),
                ghist,
                Some(gpair),
                &[node],
                None,
            ))??
            .pop()
        });
        if let Some(hist) = built {
            out.copy_from_slice(&hist);
        } else {
            self.counters.count(Strategy::Cpu, rows.len());
            CpuBackend.build(ghist, rows, gpair, out);
        }
    }

    fn prepare(&self, _ghist: &GHistIndex, gpair: &[GradPair]) {
        if let Some(mut state) = self.lock() {
            // The trainer refills its gradient buffer in place every round,
            // so a new tree always restages.
            let staged = self.stage(&mut state, gpair);
            if self.ok(staged).is_none() {
                state.staged.len = 0;
            }
        }
    }

    fn row_engine(&self) -> Option<&dyn RowEngine> {
        (!self.device.failed.load(Ordering::Acquire)).then_some(self as &dyn RowEngine)
    }
}

impl RowEngine for CudaHistBackend {
    fn begin_tree(&self, ghist: &GHistIndex, rows: &[u32]) -> Option<Segment> {
        if !self.fits(ghist) || rows.len() > self.n_rows {
            return None;
        }
        // One parallel pass: every row inside the index (the kernels do not
        // bounds-check), and whether the rows are one ascending run (then
        // generated on the device instead of uploaded).
        let first = rows.first().copied().unwrap_or(0) as usize;
        let (inside, run) = rows
            .par_chunks(1 << 16)
            .enumerate()
            .map(|(c, chunk)| {
                let base = first + (c << 16);
                let inside = chunk.iter().all(|&r| (r as usize) < self.n_rows);
                let run = chunk
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| r as usize == base + i);
                (inside, run)
            })
            .reduce(|| (true, true), |a, b| (a.0 && b.0, a.1 && b.1));
        if !inside {
            return None;
        }
        let mut state = self.lock()?;
        if state.staged.len == 0 {
            return None;
        }
        let stream = &self.device.stream;
        let placed = match run.then_some(first..first + rows.len()) {
            Some(range) if !rows.is_empty() => {
                let n = rows.len() as u64;
                let first = range.start as u32;
                let mut launch = stream.launch_builder(&self.device.kernels.iota_rows);
                launch.arg(&mut state.tree_rows).arg(&n).arg(&first);
                // SAFETY: writes `rows.len() <= n_rows` entries.
                unsafe { launch.launch(self.device.grid(rows.len())) }.map(|_| ())
            }
            _ => {
                let state = &mut *state;
                pinned(stream, &mut state.pin_rows, rows.len(), true).and_then(|staging| {
                    upload_pinned(stream, staging, rows, &mut state.tree_rows)?;
                    // The staging buffer is reused by the leaf-row download.
                    stream.synchronize()
                })
            }
        };
        self.ok(placed)?;
        state.tree_len = rows.len();
        Some(Segment {
            offset: 0,
            len: rows.len(),
        })
    }

    fn root_total(&self, seg: Segment) -> Option<GradStats> {
        let mut state = self.lock()?;
        // `sum_rows`'s blocks, each summed on the device (in integers when
        // the blocks' sums are exact, else as `f64` chains), and the block
        // totals added here in block order, the host's own operations.
        let grain = match sum_order(seg.len) {
            SumOrder::Blocked { grain } => grain,
            SumOrder::Chain => seg.len.max(1),
        };
        if !state.staged.finite() || seg.offset + seg.len > state.tree_len {
            return None;
        }
        let exact = state.staged.sums_exact(grain);
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let chunks = seg.len.div_ceil(grain).max(1);
        let blocks = (|| {
            let config = LaunchConfig {
                grid_dim: (chunks as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let rows = state.tree_rows.slice(seg.offset..seg.offset + seg.len);
            let (n, grain64) = (seg.len as u64, grain as u64);
            let mut host = vec![0f64; chunks * 2];
            if exact {
                fit(stream, &mut state.totals, chunks * 2)?;
                stream.memset_zeros(&mut state.totals.slice_mut(..chunks * 2))?;
                if seg.len > 0 {
                    let mut launch = stream.launch_builder(&device.kernels.chunk_totals);
                    launch
                        .arg(&rows)
                        .arg(&n)
                        .arg(&grain64)
                        .arg(&state.units)
                        .arg(&mut state.totals);
                    // SAFETY: one block per chunk reads its rows (below
                    // `n_rows`) and writes its two totals.
                    unsafe { launch.launch(config) }?;
                }
                let mut units = vec![0i64; chunks * 2];
                stream.memcpy_dtoh(&state.totals.slice(..chunks * 2), &mut units)?;
                stream.synchronize()?;
                let (grad, hess) = (&state.staged.grad, &state.staged.hess);
                for (h, k) in host.chunks_mut(2).zip(units.chunks(2)) {
                    // Each block's exact sum, as its `f64` chain is.
                    h[0] = grad.value(k[0]);
                    h[1] = hess.value(k[1]);
                }
            } else {
                fit(stream, &mut state.chains, chunks * 2)?;
                let mut launch = stream.launch_builder(&device.kernels.chunk_chains);
                launch
                    .arg(&rows)
                    .arg(&n)
                    .arg(&grain64)
                    .arg(&state.gpair)
                    .arg(&mut state.chains);
                // SAFETY: one thread per chunk reads its rows' pairs and
                // writes its total.
                unsafe { launch.launch(Device::one_per(chunks)) }?;
                stream.memcpy_dtoh(&state.chains.slice(..chunks * 2), &mut host)?;
                stream.synchronize()?;
            }
            Ok(host)
        })();
        let blocks = self.ok(blocks)?;
        let mut blocks = blocks.chunks(2).map(|b| GradStats::new(b[0], b[1]));
        let mut total = blocks.next().unwrap_or_default();
        for block in blocks {
            total.add(block);
        }
        Some(total)
    }

    fn partition(&self, ghist: &GHistIndex, splits: &[RowSplit<'_>]) -> Option<Vec<Partitioned>> {
        let mut state = self.lock()?;
        let cuts = ghist.cuts();
        let valid = self.fits(ghist)
            && splits.iter().all(|s| {
                let (fs, fe) = cuts.feature_bins(s.feature as usize);
                (s.feature as usize) < self.n_cols
                    && s.seg.offset + s.seg.len <= state.tree_len
                    && match s.rule {
                        RowRule::Below(_) => true,
                        RowRule::Table(table) => table.len() == fe - fs,
                    }
            });
        if !valid || u32::try_from(splits.len()).is_err() {
            return None;
        }
        if splits.is_empty() {
            return Some(Vec::new());
        }
        let parts = self.partition_on(&mut state, splits);
        self.ok(parts)
    }

    fn histograms(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[Segment],
    ) -> Option<Vec<Histogram>> {
        let mut state = self.lock()?;
        if !self.fits(ghist)
            || !state.staged.holds(gpair)
            || nodes.iter().any(|s| s.offset + s.len > state.tree_len)
        {
            return None;
        }
        if nodes.is_empty() {
            return Some(Vec::new());
        }
        let hists = self.histograms_on(&mut state, RowSource::Tree, ghist, gpair, nodes, None);
        self.ok(hists)?
    }

    fn rows(&self, segs: &[Segment]) -> Option<Vec<Vec<u32>>> {
        let mut state = self.lock()?;
        let end = segs.iter().map(|s| s.offset + s.len).max().unwrap_or(0);
        if end > state.tree_len {
            return None;
        }
        let stream = &self.device.stream;
        let state = &mut *state;
        let all = pinned(stream, &mut state.pin_rows, end, false)
            .and_then(|staging| download_pinned(stream, staging, &state.tree_rows, end));
        let all = self.ok(all)?;
        Some(
            segs.par_iter()
                .map(|s| all[s.offset..s.offset + s.len].to_vec())
                .collect(),
        )
    }

    fn load_margins(&self, margins: &[f32]) -> Option<()> {
        let mut state = self.lock()?;
        if margins.len() != self.n_rows {
            return None;
        }
        let stream = &self.device.stream;
        let loaded = stream.clone_htod(margins);
        state.margins = Some(self.ok(loaded)?);
        Some(())
    }

    fn gradients(&self, loss: DeviceLoss, labels: &[f32], weights: Option<&[f32]>) -> Option<bool> {
        let mut state = self.lock()?;
        let n = self.n_rows;
        if labels.len() != n || weights.is_some_and(|w| w.len() != n) || state.margins.is_none() {
            return None;
        }
        if let DeviceLoss::Logistic { split, .. } = loss
            && (split.lanes == 0 || split.rows > n || !split.rows.is_multiple_of(split.lanes))
        {
            return None;
        }
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let staged = (|| {
            state.staged.len = 0;
            // Labels and weights are copied once per run (keyed by the
            // host slice they came from).
            let key = |s: &[f32]| s.as_ptr().addr();
            if state.labels.as_ref().is_none_or(|(k, _)| *k != key(labels)) {
                state.labels = Some((key(labels), stream.clone_htod(labels)?));
            }
            if let Some(weights) = weights
                && state
                    .weights
                    .as_ref()
                    .is_none_or(|(k, _)| *k != key(weights))
            {
                state.weights = Some((key(weights), stream.clone_htod(weights)?));
            }
            let (Some(margins), Some((_, dev_labels))) = (&state.margins, &state.labels) else {
                return Ok(None);
            };
            let dev_weights = match (&state.weights, weights) {
                (Some((_, w)), Some(_)) => w,
                // Unread without weights; any valid pointer.
                _ => dev_labels,
            };
            let weighted = i32::from(weights.is_some());
            match loss {
                DeviceLoss::SquaredError { scale_pos_weight } => {
                    let rows = n as u64;
                    let mut launch = stream.launch_builder(&device.kernels.squared_error);
                    launch
                        .arg(margins)
                        .arg(dev_labels)
                        .arg(dev_weights)
                        .arg(&weighted)
                        .arg(&scale_pos_weight)
                        .arg(&rows)
                        .arg(&mut state.gpair);
                    // SAFETY: reads `n_rows` margins, labels and weights,
                    // and writes `n_rows` pairs.
                    unsafe { launch.launch(device.grid(n)) }?;
                }
                DeviceLoss::Logistic {
                    scale_pos_weight,
                    min_hess,
                    split,
                } => {
                    // The rows after the vectors (fewer than `lanes`, or a
                    // batch too short for vectors) run the host's scalar
                    // path, here on the host.
                    if split.rows < n {
                        let mut tail = vec![0f32; n - split.rows];
                        stream.memcpy_dtoh(&margins.slice(split.rows..n), &mut tail)?;
                        stream.synchronize()?;
                        let mut pairs = vec![GradPair::default(); tail.len()];
                        crate::simd::logistic_gradient(
                            &tail,
                            &labels[split.rows..],
                            weights.map(|w| &w[split.rows..]),
                            scale_pos_weight,
                            min_hess,
                            &mut pairs,
                        );
                        let flat: Vec<f32> = pairs.iter().flat_map(|p| [p.grad, p.hess]).collect();
                        stream.memcpy_htod(
                            &flat,
                            &mut state.gpair.slice_mut(2 * split.rows..2 * n),
                        )?;
                    }
                    if split.rows > 0 {
                        let rows = split.rows as u64;
                        let lanes = split.lanes as u32;
                        let mut launch = stream.launch_builder(&device.kernels.logistic);
                        launch
                            .arg(margins)
                            .arg(dev_labels)
                            .arg(dev_weights)
                            .arg(&weighted)
                            .arg(&scale_pos_weight)
                            .arg(&min_hess)
                            .arg(&split.max_input)
                            .arg(&lanes)
                            .arg(&rows)
                            .arg(&mut state.gpair);
                        // SAFETY: `split.rows <= n_rows` is a whole number
                        // of vectors, so every vector's margins are read
                        // inside the buffer; writes the first `split.rows`
                        // pairs.
                        unsafe { launch.launch(device.grid(split.rows)) }?;
                    }
                }
            }
            let (grad, hess) = self.device_domains(state)?;
            let finite = grad.is_finite() && hess.is_finite();
            self.stage_units(state, grad, hess)?;
            Ok(Some(finite))
        })();
        self.ok(staged)?
    }

    fn add_leaf_values(&self, leaves: &[(Segment, f32)]) -> Option<()> {
        let mut state = self.lock()?;
        if leaves
            .iter()
            .any(|(s, _)| s.offset + s.len > state.tree_len)
            || state.margins.is_none()
        {
            return None;
        }
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let mut segs = Vec::with_capacity(leaves.len() * 2);
        let mut values = Vec::with_capacity(leaves.len());
        let mut ptiles = Vec::new();
        for (k, (seg, value)) in leaves.iter().enumerate() {
            segs.extend([seg.offset as u64, seg.len as u64]);
            values.push(*value);
            ptiles.extend((0..seg.len.div_ceil(PART_TILE)).map(|t| (k as u64) << 32 | t as u64));
        }
        if ptiles.is_empty() {
            return Some(());
        }
        let added = (|| {
            upload(stream, &mut state.segs, &segs)?;
            upload(stream, &mut state.values, &values)?;
            upload(stream, &mut state.ptiles, &ptiles)?;
            let Some(margins) = state.margins.as_mut() else {
                return Ok(());
            };
            let mut launch = stream.launch_builder(&device.kernels.add_leaves);
            launch
                .arg(&state.segs)
                .arg(&state.values)
                .arg(&state.ptiles)
                .arg(&state.tree_rows)
                .arg(margins);
            let config = LaunchConfig {
                grid_dim: (ptiles.len() as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one block per tile of a leaf segment inside the tree's
            // rows; each row (below `n_rows`) is in exactly one leaf.
            unsafe { launch.launch(config) }.map(|_| ())
        })();
        self.ok(added)
    }

    fn read_margins(&self, out: &mut [f32]) -> Option<()> {
        let state = self.lock()?;
        let margins = state.margins.as_ref()?;
        if out.len() != self.n_rows {
            return None;
        }
        let stream = &self.device.stream;
        let read = stream
            .memcpy_dtoh(margins, out)
            .and_then(|()| stream.synchronize());
        self.ok(read)
    }

    fn reserve_hists(&self, ghist: &GHistIndex, slots: usize) -> Option<bool> {
        let mut state = self.lock()?;
        if !self.fits(ghist) {
            return None;
        }
        if state.pool_slots >= slots {
            return Some(true);
        }
        let words = slots.checked_mul(self.total_bins.checked_mul(2)?)?;
        let stream = &self.device.stream;
        let state = &mut *state;
        let reserved = (|| {
            stream.context().bind_to_thread()?;
            let (free, _) = cudarc::driver::result::mem_get_info()?;
            // Half of what is free (plus the pool being replaced), so the
            // per-level buffers keep room to grow.
            let held = state.pool.len() * 8;
            if words.saturating_mul(8) > free / 2 + held {
                return Ok(false);
            }
            state.pool_slots = 0;
            state.pool = stream.alloc_zeros(2)?;
            state.pool = stream.alloc_zeros(words)?;
            state.pool_slots = slots;
            Ok(true)
        })();
        self.ok(reserved)
    }

    fn build_resident(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[(Segment, HistSlot)],
        siblings: &[(HistSlot, HistSlot)],
    ) -> Option<()> {
        let mut state = self.lock()?;
        let slots = state.pool_slots;
        let in_pool = |slot: HistSlot| (slot as usize) < slots;
        if !self.fits(ghist)
            || !state.staged.holds(gpair)
            || nodes
                .iter()
                .any(|&(s, slot)| s.offset + s.len > state.tree_len || !in_pool(slot))
            || siblings
                .iter()
                .any(|&(parent, built)| parent == built || !in_pool(parent) || !in_pool(built))
        {
            return None;
        }
        let segs: Vec<Segment> = nodes.iter().map(|&(seg, _)| seg).collect();
        let targets: Vec<HistSlot> = nodes.iter().map(|&(_, slot)| slot).collect();
        let state = &mut *state;
        // The build writes `state.out`: make it the pool for this call.
        std::mem::swap(&mut state.out, &mut state.pool);
        let built = self.histograms_on(state, RowSource::Tree, ghist, gpair, &segs, Some(&targets));
        std::mem::swap(&mut state.out, &mut state.pool);
        // `None`: CPU-built nodes without the host gradients.
        self.ok(built)??;
        if siblings.is_empty() {
            return Some(());
        }
        let device = &*self.device;
        let stream = &device.stream;
        let subtracted = (|| {
            let pairs: Vec<u32> = siblings
                .iter()
                .flat_map(|&(parent, built)| [parent, built])
                .collect();
            upload(stream, &mut state.nodes, &pairs)?;
            let (n_pairs, total_bins) = (siblings.len() as u64, self.total_bins as u64);
            let mut launch = stream.launch_builder(&device.kernels.subtract_hists);
            launch
                .arg(&mut state.pool)
                .arg(&state.nodes)
                .arg(&n_pairs)
                .arg(&total_bins);
            // SAFETY: every pair names two distinct slots of the pool, and
            // each element writes only its own parent bin.
            unsafe { launch.launch(device.grid(siblings.len() * self.total_bins)) }.map(|_| ())
        })();
        self.ok(subtracted)
    }

    fn scan_resident(
        &self,
        ghist: &GHistIndex,
        reg: &RegParams,
        requests: &[ScanRequest<'_>],
    ) -> Option<Vec<FeatureScan>> {
        let mut state = self.lock()?;
        if !self.fits(ghist)
            || requests.iter().any(|r| {
                r.slot as usize >= state.pool_slots
                    || r.features.iter().any(|&(f, _)| f as usize >= self.n_cols)
            })
        {
            return None;
        }
        let mut tasks = Vec::new();
        let mut totals = Vec::with_capacity(requests.len() * 2);
        let mut params = Vec::with_capacity(requests.len() * 3);
        for (i, r) in requests.iter().enumerate() {
            totals.extend([r.total.grad, r.total.hess]);
            params.extend([r.root_gain, r.lower, r.upper]);
            for &(feature, dir) in r.features {
                tasks.extend([i as u32, feature, r.slot, i32::from(dir) as u32]);
            }
        }
        let n_tasks = tasks.len() / 4;
        if n_tasks == 0 {
            return Some(Vec::new());
        }
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let scanned = (|| {
            upload(stream, &mut state.scan_tasks, &tasks)?;
            upload(stream, &mut state.scan_totals, &totals)?;
            upload(stream, &mut state.scan_params, &params)?;
            fit(stream, &mut state.scan_meta, n_tasks * 4)?;
            fit(stream, &mut state.scan_acc, n_tasks * 2)?;
            let (total_bins, n) = (self.total_bins as u64, n_tasks as u64);
            let dense = i32::from(self.dense);
            let mut launch = stream.launch_builder(&device.kernels.scan_splits);
            launch
                .arg(&state.pool)
                .arg(&state.feature_first)
                .arg(&total_bins)
                .arg(&state.scan_tasks)
                .arg(&n)
                .arg(&state.scan_totals)
                .arg(&state.scan_params)
                .arg(&reg.lambda)
                .arg(&reg.alpha)
                .arg(&reg.max_delta_step)
                .arg(&reg.min_child_weight)
                .arg(&dense)
                .arg(&mut state.scan_meta)
                .arg(&mut state.scan_acc);
            let config = LaunchConfig {
                grid_dim: (n_tasks.div_ceil(SCAN_WARPS) as u32, 1, 1),
                block_dim: (32 * SCAN_WARPS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one warp per task; each names a request, a feature
            // below `n_cols` (whose bins `feature_first` bounds) and a pool
            // slot, and writes its own result words.
            unsafe { launch.launch(config) }?;
            let mut meta = vec![0u32; n_tasks * 4];
            let mut acc = vec![0f64; n_tasks * 2];
            stream.memcpy_dtoh(&state.scan_meta.slice(..meta.len()), &mut meta)?;
            stream.memcpy_dtoh(&state.scan_acc.slice(..acc.len()), &mut acc)?;
            stream.synchronize()?;
            Ok((meta, acc))
        })();
        let (meta, acc) = self.ok(scanned)?;
        Some(
            meta.as_chunks::<4>()
                .0
                .iter()
                .zip(acc.as_chunks::<2>().0)
                .map(|(m, a)| match m[0] {
                    0 => FeatureScan::Empty,
                    1 => FeatureScan::Best {
                        loss_chg: f32::from_bits(m[2]),
                        backward: m[3] != 0,
                        offset: m[1],
                        acc: GradStats::new(a[0], a[1]),
                    },
                    _ => FeatureScan::Nan,
                })
                .collect(),
        )
    }

    fn read_hist(&self, slot: HistSlot) -> Option<Histogram> {
        let state = self.lock()?;
        let s = slot as usize;
        if s >= state.pool_slots {
            return None;
        }
        let bins = self.total_bins;
        let mut flat = vec![0f64; bins * 2];
        let stream = &self.device.stream;
        let read = stream
            .memcpy_dtoh(
                &state.pool.slice(s * bins * 2..(s + 1) * bins * 2),
                &mut flat,
            )
            .and_then(|()| stream.synchronize());
        self.ok(read)?;
        Some(
            flat.as_chunks::<2>()
                .0
                .iter()
                .map(|&[grad, hess]| GradStats::new(grad, hess))
                .collect(),
        )
    }
}

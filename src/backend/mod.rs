//! Optional compute backends.
//!
//! Training and prediction run on the CPU by default, exactly as they always
//! have. This module holds the opt-in accelerators:
//!
//! - [`metal`] (macOS only, `metal` feature): native Metal GPU acceleration
//!   for histogram construction during training (`device = metal`) and for
//!   batch prediction ([`GpuModel`](metal::GpuModel), from
//!   `BoostedModel::to_gpu`).
//! - [`cuda`] (Linux only, `cuda` feature): NVIDIA GPU histogram
//!   construction during training (`device = cuda`).
//!
//! The backends keep the crate's determinism contract: a GPU run reproduces
//! the CPU result bit for bit (work the GPU cannot compute exactly runs on
//! the CPU; see [`metal`] and [`cuda`]), and repeats itself exactly
//! across runs and machines.
//!
//! A future `wgpu` backend will extend the same seam to Windows.

/// When a GPU backend's integer histogram sums reproduce the CPU's `f64`
/// sums (platform-independent, so its proof is tested everywhere).
#[cfg_attr(
    not(any(
        all(target_os = "macos", feature = "metal"),
        all(target_os = "linux", feature = "cuda")
    )),
    allow(
        dead_code,
        reason = "only the GPU backends call it; its unit tests run on every platform"
    )
)]
mod exact_sum;

/// The CUDA backend (Linux, `cuda` feature).
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod cuda;

/// The CUDA backend's stand-in when it is not compiled in (any other
/// platform, or the feature off): the module exists so `backend::cuda`
/// paths and doc links resolve on every platform. `device = cuda` is then
/// refused by [`TrainingParams::validate`](crate::config::TrainingParams::validate).
///
/// The backend's design, exactness rules, and limitations are documented
/// in the real module: run `cargo doc --features cuda --open` on Linux.
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
pub mod cuda {}

/// The native Metal backend (macOS, `metal` feature).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

/// The Metal backend's stand-in when it is not compiled in (any other
/// platform, or the feature off): the module exists so `backend::metal`
/// paths and doc links resolve on every platform, but holds only the
/// [`GpuModel`](self::metal::GpuModel) handle, which
/// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then never
/// constructs — it always returns an error naming the missing feature.
///
/// These docs are the stand-in (docs.rs builds on Linux). The Metal API and
/// the backend's design, exactness bound, and limitations are documented
/// in the real module: run `cargo doc --features metal --open` on macOS.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub mod metal {
    /// The GPU predictor handle when the Metal backend is not compiled in.
    /// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu) then
    /// always returns an error, so this is never constructed.
    #[derive(Debug)]
    #[non_exhaustive]
    pub struct GpuModel;
}

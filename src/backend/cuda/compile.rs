//! Run-time kernel compilation: NVRTC straight to CUBIN for one
//! architecture, so the driver never JIT-compiles PTX (which would fail
//! when the installed NVRTC is newer than the driver).

use crate::error::{HessboostError, Result};
use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::{Ptx, result, sys};
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// The kernel source (`kernels.cu`).
const SOURCE: &str = include_str!("kernels.cu");

/// NVRTC options besides the architecture. No FP contraction, no
/// flush-to-zero, IEEE division and square root: every floating-point
/// operation is the one IEEE operation the CPU performs. `-lineinfo` keeps
/// source lines for Nsight Compute and `compute-sanitizer`.
const OPTIONS: &[&str] = &[
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--std=c++17",
    "-lineinfo",
];

/// An NVRTC program, destroyed on drop.
struct Program(sys::nvrtcProgram);

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: the program was created by `create_program` and is
        // destroyed only here.
        let _ = unsafe { result::destroy_program(self.0) };
    }
}

/// Compile the kernels to a CUBIN for `arch` (`sm_89`, ...), or the
/// compiler's error and log. NVRTC must be loadable (checked by the
/// caller).
fn cubin(arch: &str) -> std::result::Result<Vec<u8>, String> {
    let source = CString::new(SOURCE).map_err(|e| e.to_string())?;
    let program = Program(
        result::create_program(&source, Some(c"hessboost_kernels.cu"))
            .map_err(|e| format!("NVRTC program: {e}"))?,
    );
    let mut options = vec![format!("--gpu-architecture={arch}")];
    options.extend(OPTIONS.iter().map(|o| (*o).to_owned()));
    // SAFETY: `program` is a live program created above.
    if let Err(error) = unsafe { result::compile_program(program.0, &options) } {
        // SAFETY: as above.
        let log = unsafe { result::get_program_log(program.0) }
            .ok()
            .map(|log| {
                // SAFETY: NVRTC returns the log NUL-terminated.
                unsafe { CStr::from_ptr(log.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        return Err(format!(
            "NVRTC failed to compile the kernels for {arch}: {error}\n{log}"
        ));
    }
    let mut size = 0usize;
    // SAFETY: `program` compiled successfully; the call writes the size.
    unsafe { sys::nvrtcGetCUBINSize(program.0, &raw mut size) }
        .result()
        .map_err(|e| format!("NVRTC CUBIN size: {e}"))?;
    let mut image = vec![0u8; size];
    // SAFETY: `image` holds the `size` bytes NVRTC reported.
    unsafe { sys::nvrtcGetCUBIN(program.0, image.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| format!("NVRTC CUBIN: {e}"))?;
    Ok(image)
}

/// Compile the kernels for `arch` and load them into `ctx`.
pub(super) fn load(
    ctx: &Arc<CudaContext>,
    arch: &str,
) -> std::result::Result<Arc<CudaModule>, String> {
    let image = cubin(arch)?;
    ctx.load_module(Ptx::from_binary(image))
        .map_err(|e| format!("CUDA module load for {arch}: {e}"))
}

/// Compile the CUDA kernels for `arch` (for example `sm_89`) without a
/// GPU, returning the CUBIN's size: the guard test that a kernel change
/// still compiles on machines with NVRTC but no device. Errors when NVRTC
/// is not loadable or the compile fails.
pub fn compile_kernels(arch: &str) -> Result<usize> {
    // SAFETY: only tries to open the library by name.
    if !unsafe { sys::is_culib_present() } {
        return Err(HessboostError::gpu("libnvrtc not found"));
    }
    cubin(arch)
        .map(|image| image.len())
        .map_err(HessboostError::gpu)
}

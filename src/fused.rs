//! A fused GeGLU kernel for the Metal backend.
//!
//! ModernBERT's MLP computes `gelu(gate) * up` where gate and up are the two
//! halves of one wide `Wi` projection. Done with candle ops that is two passes
//! over a [tokens, intermediate] tensor — gelu writes it, the multiply reads it
//! back. This kernel does both in one pass, reading gate and up straight out of
//! the wide `[tokens, 2*intermediate]` projection so no chunk copy is needed
//! either.
//!
//! It is a [`CustomOp1`] with candle unmodified: the shader is compiled at
//! runtime through candle's public Metal wrappers and dispatched onto candle's
//! command buffer, exactly as candle's own `UgIOp1` does. Non-Metal callers use
//! [`geglu`], which falls back to the plain candle path.

use candle_core::{Result, Tensor};

/// `gelu(gate) * up`, where `wide` is `[.., 2 * inter]` with gate in the first
/// `inter` columns and up in the rest. Returns `[.., inter]`.
///
/// On Metal this runs the fused kernel; elsewhere it splits and uses candle's
/// ops, so the same call works on every backend. The MLP only reaches this on
/// Metal (the CPU keeps a pre-split Wi), but the fallback keeps it total.
pub fn geglu(wide: &Tensor, inter: usize) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if wide.device().is_metal() && wide.dtype() == candle_core::DType::F32 {
        // The kernel works on a 2-D [rows, 2*inter]; flatten any leading dims
        // (a bucketed batch arrives as [b, seq, 2*inter]) and restore them.
        let dims = wide.dims();
        let (lead, cols) = dims.split_at(dims.len() - 1);
        let rows: usize = lead.iter().product();
        let flat = wide.reshape((rows, cols[0]))?;
        let out = flat.apply_op1_no_bwd(&metal::GegluWide { inter })?;
        let mut out_shape = lead.to_vec();
        out_shape.push(inter);
        return out.reshape(out_shape);
    }
    let gate = wide.narrow(candle_core::D::Minus1, 0, inter)?;
    let up = wide.narrow(candle_core::D::Minus1, inter, inter)?;
    gate.gelu_erf()? * up
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::backend::BackendStorage;
    use candle_core::{CustomOp1, Layout, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputePipeline, Device};
    use std::sync::{Mutex, OnceLock};

    // erf is candle's own A&S 7.1.26 implementation, so gelu_erf(gate) matches
    // the split path's arithmetic rather than only to a tolerance.
    const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float kohagi_erf(float in) {
    constexpr const float a1 =  0.254829592;
    constexpr const float a2 = -0.284496736;
    constexpr const float a3 =  1.421413741;
    constexpr const float a4 = -1.453152027;
    constexpr const float a5 =  1.061405429;
    constexpr const float p  =  0.3275911;
    float x = in;
    int sign = 1;
    if (x < 0) sign = -1;
    x = fabs(x);
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    return sign*y;
}

// wide is [M, 2I] row-major. Row r's gate is wide[r*2I + c], up is
// wide[r*2I + I + c], for c in [0, I). One thread per output element.
kernel void geglu_wide_f32(
    device const float *wide [[buffer(0)]],
    device float       *out  [[buffer(1)]],
    constant uint      &m    [[buffer(2)]],
    constant uint      &i    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= m * i) return;
    uint row = gid / i;
    uint col = gid % i;
    uint base = row * 2u * i;
    float g = wide[base + col];
    float u = wide[base + i + col];
    float gelu = g * (1.0f + kohagi_erf(g * M_SQRT1_2_F)) / 2.0f;
    out[gid] = gelu * u;
}
"#;

    /// Cached pipeline per Metal device. candle's own kernel cache is keyed by a
    /// closed enum we cannot extend, so we hold our own. One shader, one dtype,
    /// so the device is the only key that matters.
    fn pipeline(dev: &Device) -> Result<ComputePipeline> {
        static CACHE: OnceLock<Mutex<Vec<(usize, ComputePipeline)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = dev.registry_id() as usize;
        let mut guard = cache.lock().unwrap();
        if let Some((_, p)) = guard.iter().find(|(k, _)| *k == key) {
            return Ok(p.clone());
        }
        // Safe math rather than Metal's default fast math, so the compiler does
        // not reorder or approximate the float ops. It measured no different
        // here — the fused kernel moves the Metal output by 1.5e-13 against the
        // split path — but it keeps that true if the shader grows, which matters
        // for Kohagi's "f32 is f32 everywhere" claim.
        let opts = objc2_metal::MTLCompileOptions::new();
        opts.setMathMode(objc2_metal::MTLMathMode::Safe);
        let lib = dev
            .new_library_with_source(SHADER, Some(&opts))
            .map_err(candle_core::Error::wrap)?;
        let func = lib
            .get_function("geglu_wide_f32", None)
            .map_err(candle_core::Error::wrap)?;
        let pipe = dev
            .new_compute_pipeline_state_with_function(&func)
            .map_err(candle_core::Error::wrap)?;
        guard.push((key, pipe.clone()));
        Ok(pipe)
    }

    pub struct GegluWide {
        pub inter: usize,
    }

    impl CustomOp1 for GegluWide {
        fn name(&self) -> &'static str {
            "geglu_wide"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            // geglu() never routes the CPU here, but the trait requires it.
            candle_core::bail!("geglu_wide is metal-only; use the split path on cpu")
        }

        fn metal_fwd(&self, wide: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
            if !l.is_contiguous() {
                candle_core::bail!("geglu_wide needs a contiguous input");
            }
            let (rows, cols) = l.shape().dims2()?;
            if cols != 2 * self.inter {
                candle_core::bail!(
                    "geglu_wide: expected [.., {}], got [.., {cols}]",
                    2 * self.inter
                );
            }
            let n = rows * self.inter;
            let device = wide.device();
            let out = device
                .new_buffer_builder()
                .with_size_for(n, candle_core::DType::F32)
                .with_label("geglu_wide")
                .build()?;

            let pipe = pipeline(device.metal_device())?;
            let encoder = device.command_encoder()?;
            let enc = encoder.as_ref();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_input_buffer(0, Some(wide.buffer()), l.start_offset() * 4);
            enc.set_output_buffer(1, Some(&out), 0);
            let rows32 = rows as u32;
            let inter32 = self.inter as u32;
            enc.set_bytes_directly(2, 4, &rows32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes_directly(3, 4, &inter32 as *const u32 as *const std::ffi::c_void);
            let tew = pipe.max_total_threads_per_threadgroup().min(256) as usize;
            enc.dispatch_threads(
                objc2_metal::MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: tew,
                    height: 1,
                    depth: 1,
                },
            );

            Ok((
                MetalStorage::new(out, device.clone(), n, candle_core::DType::F32),
                (rows, self.inter).into(),
            ))
        }
    }
}

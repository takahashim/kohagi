//! A fused gated-feed-forward kernel for the Metal backend.
//!
//! ModernBERT's MLP computes `act(gate) * up` where gate and up are the two
//! halves of one wide `Wi` projection, and `act` is the gelu or silu the config
//! names (a GeGLU or a SwiGLU). Done with candle ops that is two passes
//! over a [tokens, intermediate] tensor — gelu writes it, the multiply reads it
//! back. This kernel does both in one pass, reading gate and up straight out of
//! the wide `[tokens, 2*intermediate]` projection so no chunk copy is needed
//! either.
//!
//! It is a [`CustomOp1`] with candle unmodified: the shader is compiled at
//! runtime through candle's public Metal wrappers and dispatched onto candle's
//! command buffer, exactly as candle's own `UgIOp1` does. Non-Metal callers use
//! [`gated`], which falls back to the plain candle path.

use candle_core::{Result, Tensor};

use crate::encoder::Activation;

/// `act(gate) * up`, where `wide` is `[.., 2 * inter]` with gate in the first
/// `inter` columns and up in the rest. Returns `[.., inter]`.
///
/// On Metal this runs the fused kernel; elsewhere it splits and uses candle's
/// ops, so the same call works on every backend. The MLP only reaches this on
/// Metal (the CPU keeps a pre-split Wi), but the fallback keeps it total.
pub fn gated(wide: &Tensor, inter: usize, act: Activation) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if wide.device().is_metal() && wide.dtype() == candle_core::DType::F32 {
        // The kernel works on a 2-D [rows, 2*inter]; flatten any leading dims
        // (a bucketed batch arrives as [b, seq, 2*inter]) and restore them.
        let dims = wide.dims();
        let (lead, cols) = dims.split_at(dims.len() - 1);
        let rows: usize = lead.iter().product();
        let flat = wide.reshape((rows, cols[0]))?;
        let out = flat.apply_op1_no_bwd(&metal::GatedWide { inter, act })?;
        let mut out_shape = lead.to_vec();
        out_shape.push(inter);
        return out.reshape(out_shape);
    }
    // The CPU kernel reads the interleaved `[.., 2 * inter]` layout directly,
    // which is the whole reason `crate::bf16::geglu` carries two entry points.
    // Narrowing first would hand it two strided views and force a copy to undo.
    if act == Activation::Gelu && cpu::takes(wide) {
        return wide.apply_op1_no_bwd(&cpu::GegluWide { inter });
    }
    let gate = wide.narrow(candle_core::D::Minus1, 0, inter)?;
    let up = wide.narrow(candle_core::D::Minus1, inter, inter)?;
    gated_split(&gate, &up, act)
}

/// `act(gate) * up` from two separate tensors, which is the shape the CPU's
/// pre-split `Wi` leaves them in.
///
/// On a CPU where [`crate::bf16::geglu`] reaches a vector kernel this takes it,
/// and elsewhere it is candle's `gelu_erf` and a multiply. The kernel is worth
/// the detour for the same reason on both architectures and by very different
/// amounts: candle evaluates `erf` one element at a time, and the MLP's
/// intermediate is four times the hidden width, so this is the widest
/// elementwise op in the model. Measured at ruri-v3-130m's shape on an M2, 19
/// calls over `[460, 2048]`: 0.0600 s through `gelu_erf` against 0.0176 s
/// through the kernel.
///
/// Silu is left to candle. It has no hand-written kernel here, and inventing
/// one to match would be a second approximation to keep honest for a
/// nonlinearity none of Kohagi's models use.
pub fn gated_split(gate: &Tensor, up: &Tensor, act: Activation) -> Result<Tensor> {
    if act == Activation::Gelu && cpu::takes(gate) && cpu::takes(up) {
        return gate.apply_op2_no_bwd(up, &cpu::Geglu);
    }
    match act {
        Activation::Gelu => gate.gelu_erf()? * up,
        Activation::Silu => gate.silu()? * up,
    }
}

mod cpu {
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor};

    /// Whether the kernel can read this tensor as it stands.
    ///
    /// Contiguity is checked here rather than only inside the op so that a
    /// strided view takes candle's path instead of failing: [`super::gated`]
    /// narrows a wide projection into two views, and those are a legitimate way
    /// to arrive. The check inside each op stays as the invariant it is.
    pub fn takes(t: &Tensor) -> bool {
        crate::bf16::geglu::vectorised()
            && t.device().is_cpu()
            && t.dtype() == candle_core::DType::F32
            && t.is_contiguous()
    }

    /// `gelu(gate) * up` over one wide `[.., 2 * inter]` f32 CPU tensor, gate
    /// half first.
    pub struct GegluWide {
        pub inter: usize,
    }

    impl CustomOp1 for GegluWide {
        fn name(&self) -> &'static str {
            "kohagi-geglu-wide"
        }

        fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
            let CpuStorage::F32(v) = s else {
                candle_core::bail!("kohagi-geglu-wide takes f32, got {:?}", s.dtype());
            };
            let Some((from, to)) = l.contiguous_offsets() else {
                candle_core::bail!("kohagi-geglu-wide needs a contiguous input");
            };
            let dims = l.shape().dims();
            let (lead, cols) = dims.split_at(dims.len() - 1);
            if cols[0] != 2 * self.inter {
                candle_core::bail!(
                    "kohagi-geglu-wide wants {} columns, got {}",
                    2 * self.inter,
                    cols[0]
                );
            }
            let rows: usize = lead.iter().product();
            let out = crate::bf16::geglu::geglu(&v[from..to], rows, self.inter);
            let mut shape = lead.to_vec();
            shape.push(self.inter);
            Ok((CpuStorage::F32(out), shape.into()))
        }
    }

    /// `gelu(gate) * up` over two f32 CPU tensors of the same shape.
    ///
    /// A [`CustomOp2`] rather than slicing the tensors from outside, because
    /// candle exposes a storage buffer no other way: `to_vec1` copies, and the
    /// copy is the cost this exists to remove.
    pub struct Geglu;

    impl CustomOp2 for Geglu {
        fn name(&self) -> &'static str {
            "kohagi-geglu"
        }

        fn cpu_fwd(
            &self,
            s1: &CpuStorage,
            l1: &Layout,
            s2: &CpuStorage,
            l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let (CpuStorage::F32(a), CpuStorage::F32(b)) = (s1, s2) else {
                candle_core::bail!("kohagi-geglu takes f32, got {:?}", s1.dtype());
            };
            // The caller hands this two freshly projected tensors, so both are
            // contiguous. Saying so rather than assuming it: a strided view
            // would silently gate the wrong elements.
            let (Some(g), Some(u)) = (l1.contiguous_offsets(), l2.contiguous_offsets()) else {
                candle_core::bail!("kohagi-geglu needs contiguous inputs");
            };
            let (gate, up) = (&a[g.0..g.1], &b[u.0..u.1]);
            if gate.len() != up.len() {
                candle_core::bail!(
                    "kohagi-geglu operands differ: {} against {}",
                    gate.len(),
                    up.len()
                );
            }
            // One row of the full width: the kernel walks a flat pair of
            // buffers, and the shape it came from only decides what is returned.
            let out = crate::bf16::geglu::geglu_split(gate, up, 1, gate.len());
            Ok((CpuStorage::F32(out), l1.shape().clone()))
        }
    }
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::backend::BackendStorage;
    use candle_core::{CustomOp1, Layout, MetalStorage, Result, Shape};

    use crate::encoder::Activation;
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

// The same, with silu on the gate: x / (1 + exp(-x)), matching candle's silu so
// the fused and split paths agree by arithmetic rather than by tolerance.
kernel void swiglu_wide_f32(
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
    out[gid] = (g / (1.0f + exp(-g))) * u;
}
"#;

    /// Cached pipeline per (device, activation). candle's own kernel cache is keyed
    /// by a closed enum we cannot extend, so we hold our own. One shader and one
    /// dtype, so those two are the only keys that matter.
    fn pipeline(dev: &Device, act: Activation) -> Result<ComputePipeline> {
        /// Metal device registry id and gate activation.
        type Key = (usize, Activation);
        static CACHE: OnceLock<Mutex<Vec<(Key, ComputePipeline)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = (dev.registry_id() as usize, act);
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
        let name = match act {
            Activation::Gelu => "geglu_wide_f32",
            Activation::Silu => "swiglu_wide_f32",
        };
        let func = lib
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipe = dev
            .new_compute_pipeline_state_with_function(&func)
            .map_err(candle_core::Error::wrap)?;
        guard.push((key, pipe.clone()));
        Ok(pipe)
    }

    pub struct GatedWide {
        pub inter: usize,
        pub act: Activation,
    }

    impl CustomOp1 for GatedWide {
        fn name(&self) -> &'static str {
            "gated_wide"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            // gated() never routes the CPU here, but the trait requires it.
            candle_core::bail!("gated_wide is metal-only; use the split path on cpu")
        }

        fn metal_fwd(&self, wide: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
            if !l.is_contiguous() {
                candle_core::bail!("gated_wide needs a contiguous input");
            }
            let (rows, cols) = l.shape().dims2()?;
            if cols != 2 * self.inter {
                candle_core::bail!(
                    "gated_wide: expected [.., {}], got [.., {cols}]",
                    2 * self.inter
                );
            }
            let n = rows * self.inter;
            let device = wide.device();
            let out = device
                .new_buffer_builder()
                .with_size_for(n, candle_core::DType::F32)
                .with_label("gated_wide")
                .build()?;

            let pipe = pipeline(device.metal_device(), self.act)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The two gate activations against a plain candle split, on whatever device
    /// the build has. On Metal that exercises the fused kernels; elsewhere it
    /// checks the fallback's own dispatch.
    #[test]
    fn gated_matches_a_split_reference() {
        // Metal when the build has it, so the fused kernels are what runs; the
        // fallback otherwise.
        #[cfg(feature = "metal")]
        let device = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
        #[cfg(not(feature = "metal"))]
        let device = candle_core::Device::Cpu;
        let (rows, inter) = (4usize, 6usize);
        let values: Vec<f32> = (0..rows * 2 * inter)
            .map(|i| (i as f32 / 7.0) - 3.0)
            .collect();
        let wide = Tensor::from_vec(values.clone(), (rows, 2 * inter), &device).unwrap();

        for act in [Activation::Gelu, Activation::Silu] {
            let got = gated(&wide, inter, act).unwrap().flatten_all().unwrap();
            let got = got.to_vec1::<f32>().unwrap();
            let vals = &values;
            let want: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    (0..inter).map(move |c| {
                        let g = f64::from(vals[r * 2 * inter + c]);
                        let u = f64::from(vals[r * 2 * inter + inter + c]);
                        let a = match act {
                            // erf via Abramowitz & Stegun 7.1.26, as the shader uses.
                            Activation::Gelu => {
                                let z = g / std::f64::consts::SQRT_2;
                                let (sign, z) = (z.signum(), z.abs());
                                let t = 1.0 / (1.0 + 0.327_591_1 * z);
                                let poly = ((((1.061_405_429 * t - 1.453_152_027) * t
                                    + 1.421_413_741)
                                    * t
                                    - 0.284_496_736)
                                    * t
                                    + 0.254_829_592)
                                    * t;
                                g * 0.5 * (1.0 + sign * (1.0 - poly * (-z * z).exp()))
                            }
                            Activation::Silu => g / (1.0 + (-g).exp()),
                        };
                        (a * u) as f32
                    })
                })
                .collect();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5,
                    "{act:?} element {i}: got {g}, want {w}"
                );
            }
        }
    }
}

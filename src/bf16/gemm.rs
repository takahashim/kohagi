//! CPU bf16 GEMM built on AVX512-BF16's `VDPBF16PS` instruction.
//!
//! Computes `Y[M,N] (f32) = X[M,K] @ W[N,K]ᵀ` — the `Linear` op that dominates
//! a transformer forward — by converting both operands to bf16 and
//! accumulating the products in f32. bf16 halves the bytes moved per multiply
//! and the instruction does two multiply-accumulates per lane, which is where
//! the ~2× over candle's f32 gemm comes from. Accumulation stays f32, so the
//! precision loss is only in the inputs (cosine ≈ 0.999 per Linear, and the
//! model-level error is smaller still — see `super`).
//!
//! Three details make it fast:
//!
//! - **Broadcast form.** Each `_mm512_dpbf16_ps` broadcasts one activation
//!   K-pair against 16 weight columns, so results accumulate in the N lanes
//!   and never need a horizontal sum at the end. The obvious dot-product form
//!   (one output per lane, hsum to finish) measured only ~1.2× f32.
//! - **Register tiling.** [`MR`] rows × [`NB`] 16-wide column vectors are kept
//!   in 16 accumulator registers across the whole K loop.
//! - **Weights packed once.** [`pack_w_vnni`] rearranges a weight matrix into
//!   the interleaved layout `_mm512_dpbf16_ps` expects, at load time. Only the
//!   activations are packed per forward pass.

use half::bf16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Rows per register tile.
pub const MR: usize = 4;
/// 16-wide column vectors per register tile.
pub const NB: usize = 4;
/// f32 lanes in a 512-bit register.
pub const NLANE: usize = 16;
/// Columns per register tile; N is padded up to a multiple of this.
pub const NTILE: usize = NLANE * NB;
/// Column-panel width kept resident in L2 and reused across a band's row tiles.
pub const NC_COLS: usize = 256;

/// Whether this CPU has the instructions the kernel needs.
pub fn supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Convert activations `X[rows, cols]` to row-major bf16, padding columns to a
/// multiple of 32 and rows to a multiple of [`MR`] so the kernel can read full
/// tiles without bounds checks. Padding is zero, which contributes nothing to
/// the dot products.
pub fn pack_x(src: &[f32], rows: usize, cols: usize) -> (Vec<u16>, usize, usize) {
    let kpad = cols.div_ceil(32) * 32;
    let rpad = rows.div_ceil(MR) * MR;
    let mut out = vec![0u16; rpad * kpad];
    for r in 0..rows {
        let s = &src[r * cols..r * cols + cols];
        let d = &mut out[r * kpad..r * kpad + cols];
        #[cfg(target_arch = "x86_64")]
        if supported() {
            // SAFETY: guarded by the same feature check the kernel uses.
            unsafe { pack_row_avx512(s.as_ptr(), d.as_mut_ptr(), cols) };
            continue;
        }
        for c in 0..cols {
            d[c] = bf16::from_f32(s[c]).to_bits();
        }
    }
    (out, rpad, kpad)
}

/// Vectorized f32→bf16 of one contiguous row (round-to-nearest-even),
/// preserving element order so K indices line up with the packed weights.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn pack_row_avx512(src: *const f32, dst: *mut u16, cols: usize) {
    let mut c = 0;
    while c + 32 <= cols {
        let a = _mm512_loadu_ps(src.add(c)); // elements c..c+15
        let b = _mm512_loadu_ps(src.add(c + 16)); // elements c+16..c+31
        // cvtne2ps_pbh(hi, lo) puts `lo` in the low 16 bf16 lanes and `hi` in
        // the high 16 — i.e. back in the original element order.
        let packed = _mm512_cvtne2ps_pbh(b, a);
        _mm512_storeu_si512(
            dst.add(c) as *mut __m512i,
            std::mem::transmute::<__m512bh, __m512i>(packed),
        );
        c += 32;
    }
    while c < cols {
        *dst.add(c) = bf16::from_f32(*src.add(c)).to_bits();
        c += 1;
    }
}

/// Convert a weight matrix `W[n, k]` (row-major, PyTorch's `Linear` layout)
/// into the interleaved bf16 layout the kernel loads.
///
/// Weights are grouped into blocks of 16 columns; within a block, consecutive
/// K pairs sit next to each other per column (`[k-pair][col 0..15][kk 0..1]`),
/// which is exactly one 512-bit load per `_mm512_dpbf16_ps` operand. `n` is
/// padded to [`NTILE`] and `k` to a multiple of 32, with zeros.
pub fn pack_w_vnni(w: &[f32], n: usize, k: usize) -> (Vec<u16>, usize, usize) {
    let kpad = k.div_ceil(32) * 32;
    let npad = n.div_ceil(NTILE) * NTILE;
    let kpairs = kpad / 2;
    let mut out = vec![0u16; npad * kpad];
    for nb in 0..npad / NLANE {
        let block = &mut out[nb * kpairs * 32..(nb + 1) * kpairs * 32];
        for kp in 0..kpairs {
            for col in 0..NLANE {
                let nrow = nb * NLANE + col;
                for kk in 0..2 {
                    let kidx = 2 * kp + kk;
                    let v = if nrow < n && kidx < k {
                        bf16::from_f32(w[nrow * k + kidx]).to_bits()
                    } else {
                        0
                    };
                    block[kp * 32 + col * 2 + kk] = v;
                }
            }
        }
    }
    (out, npad, kpad)
}

/// One register tile: [`MR`] rows × [`NB`]×16 columns, accumulated over all K.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn micro(xb: *const u16, wblk: *const u16, kpad: usize, out: *mut f32, out_rs: usize) {
    /// Broadcast one row's K-pair to all 16 lanes.
    #[inline(always)]
    unsafe fn bcast(xb: *const u16, row: usize, kpad: usize, kp: usize) -> __m512bh {
        let p = xb.add(row * kpad + 2 * kp) as *const i32;
        std::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(p.read_unaligned()))
    }
    /// Load 16 columns × 1 K-pair of packed weights.
    #[inline(always)]
    unsafe fn loadb(wblk: *const u16, nv: usize, kpairs: usize, kp: usize) -> __m512bh {
        let p = wblk.add(nv * kpairs * 32 + kp * 32);
        std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(p as *const __m512i))
    }
    let kpairs = kpad / 2;
    let mut acc = [[_mm512_setzero_ps(); NB]; MR];
    for kp in 0..kpairs {
        let a = [
            bcast(xb, 0, kpad, kp),
            bcast(xb, 1, kpad, kp),
            bcast(xb, 2, kpad, kp),
            bcast(xb, 3, kpad, kp),
        ];
        let b = [
            loadb(wblk, 0, kpairs, kp),
            loadb(wblk, 1, kpairs, kp),
            loadb(wblk, 2, kpairs, kp),
            loadb(wblk, 3, kpairs, kp),
        ];
        for (ti, row) in acc.iter_mut().enumerate() {
            for (nv, v) in row.iter_mut().enumerate() {
                *v = _mm512_dpbf16_ps(*v, a[ti], b[nv]);
            }
        }
    }
    for (ti, row) in acc.iter().enumerate() {
        for (nv, v) in row.iter().enumerate() {
            _mm512_storeu_ps(out.add(ti * out_rs + nv * NLANE), *v);
        }
    }
}

/// `Y[mpad, npad] = Xb @ (packed W)ᵀ`, single-threaded.
///
/// Single-threaded on purpose: callers fan whole batches across cores, so a
/// fork-join inside every Linear would add a barrier per layer and dominate
/// these per-batch shapes. N is walked in L2-resident panels of [`NC_COLS`]
/// columns, each reused across all row tiles before moving on.
///
/// Panics unless [`supported()`]; check once at model load.
pub fn gemm_st(xb: &[u16], wvnni: &[u16], mpad: usize, npad: usize, kpad: usize) -> Vec<f32> {
    assert!(mpad.is_multiple_of(MR) && npad.is_multiple_of(NTILE));
    let kpairs = kpad / 2;
    let blk = kpairs * 32;
    let mut y = vec![0f32; mpad * npad];
    let total_blocks = npad / NLANE;
    let nc_blocks = (NC_COLS / NLANE).max(NB);
    let mtiles = mpad / MR;
    let mut nb0 = 0;
    while nb0 < total_blocks {
        let panel_blocks = nc_blocks.min(total_blocks - nb0);
        for mt in 0..mtiles {
            let xrow = mt * MR * kpad;
            let mut g = 0;
            while g < panel_blocks {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `supported()` is checked at model load; the tile
                // bounds hold because mpad/npad/kpad are padded above.
                unsafe {
                    micro(
                        xb.as_ptr().add(xrow),
                        wvnni.as_ptr().add((nb0 + g) * blk),
                        kpad,
                        y.as_mut_ptr().add(mt * MR * npad + (nb0 + g) * NLANE),
                        npad,
                    );
                }
                g += NB;
            }
        }
        nb0 += panel_blocks;
    }
    y
}

/// A `Linear` layer (`Y = X·Wᵀ`, no bias) with its weight packed for the
/// kernel at construction time.
pub struct Bf16Linear {
    wv: Vec<u16>,
    npad: usize,
    kpad: usize,
    pub n: usize,
    pub k: usize,
}

impl Bf16Linear {
    /// `weight` is `[n, k]` row-major, as PyTorch and candle store it.
    pub fn new(weight: &[f32], n: usize, k: usize) -> Self {
        let (wv, npad, kpad) = pack_w_vnni(weight, n, k);
        Self { wv, npad, kpad, n, k }
    }

    /// `x` is `[m, k]` row-major f32 → `[m, n]` row-major f32.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let (xb, mpad, _) = pack_x(x, m, self.k);
        let ypad = gemm_st(&xb, &self.wv, mpad, self.npad, self.kpad);
        if self.npad == self.n {
            // No column padding: rows are already the right width; just drop
            // the padding rows.
            let mut y = ypad;
            y.truncate(m * self.n);
            y
        } else {
            let mut y = vec![0f32; m * self.n];
            for i in 0..m {
                y[i * self.n..(i + 1) * self.n]
                    .copy_from_slice(&ypad[i * self.npad..i * self.npad + self.n]);
            }
            y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift, so the test needs no rand dependency.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    /// f64 reference for `Y[m,n] = X @ Wᵀ`.
    fn linear_ref_f64(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += x[i * k + kk] as f64 * w[j * k + kk] as f64;
                }
                y[i * n + j] = acc as f32;
            }
        }
        y
    }

    fn min_cosine(a: &[f32], b: &[f32], m: usize, n: usize) -> f64 {
        let mut worst = 1.0;
        for i in 0..m {
            let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
            for j in 0..n {
                let (x, y) = (a[i * n + j] as f64, b[i * n + j] as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na > 0.0 && nb > 0.0 {
                worst = f64::min(worst, dot / (na.sqrt() * nb.sqrt()));
            }
        }
        worst
    }

    #[test]
    fn matches_an_f64_reference() {
        if !supported() {
            eprintln!("skipped: this CPU has no avx512bf16");
            return;
        }
        // Tile-aligned and ragged M/N/K, including the real model's shapes.
        for &(m, k, n) in &[
            (4, 512, 64),
            (7, 512, 1536),
            (130, 2048, 512),
            (64, 512, 4096),
            (33, 320, 80),
        ] {
            let x = pseudo_random(m * k, 0x11 + n as u64);
            let w = pseudo_random(n * k, 0x22 + n as u64);
            let got = Bf16Linear::new(&w, n, k).forward(&x, m);
            let want = linear_ref_f64(&x, &w, m, k, n);
            let c = min_cosine(&got, &want, m, n);
            assert!(c > 0.999, "shape {m}x{k}x{n}: cosine {c}");
        }
    }
}

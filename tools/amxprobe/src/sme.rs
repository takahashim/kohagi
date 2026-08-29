//! The AMX prototype's GEMM on SME, for M4 and later: the same packing, the
//! same 32x32 tile through a scratch buffer, the same driver — and the
//! documented ISA in place of the reverse-engineered one. Written on an M1,
//! where it assembles and does not run: `main` checks `hw.optional.arm.FEAT_SME`
//! and stops there when it is 0. Everything past that line is unverified until
//! it has been run on an M4.
//!
//! What maps onto what (M4: SVL = 512 bits, ZA = 4 KB, four 32-bit tiles):
//!
//! | AMX | SME |
//! | --- | --- |
//! | `amx set` / `amx clr` | `smstart` / `smstop` |
//! | `ldx` / `ldy` (64 bytes) | `ld1w {{ zN.s }}` (one streaming vector, 16 f32) |
//! | `fma32` (16x16 outer product into an interleaved Z tile) | `fmopa zaN.s` |
//! | `stz` (one Z row) | `st1w {{ zaNh.s[w12, #i] }}` (one ZA row slice) |
//! | init bit 27 | `zero {{ za }}` |
//!
//! And what differs: `fmopa ZA, Pn, Pm, Zn, Zm` puts `Zn` on the *rows*, so A's
//! column panel is the first vector operand, where on AMX it was Y. Loads have
//! no alignment rule. Partial tiles could use predicates instead of zero
//! padding; this keeps the padding so the two jigs stay comparable.
//!
//! Streaming mode forbids NEON, so the NEON packing runs before `smstart` and
//! each tile is one `asm!` block that enters and leaves streaming mode itself —
//! the compiler may emit NEON anywhere between blocks. Entering streaming mode
//! makes every Z and P register UNKNOWN, which is what the clobber list says.
use std::arch::asm;
use std::os::raw::{c_float, c_int};
use std::time::Instant;

extern "C" {
    fn sysctlbyname(name: *const u8, old: *mut u8, oldlen: *mut usize, new: *mut u8, newlen: usize) -> i32;
    fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
}
fn has_sme() -> bool {
    let mut v: u32 = 0; let mut len = 4usize;
    unsafe { sysctlbyname(b"hw.optional.arm.FEAT_SME\0".as_ptr(), &mut v as *mut u32 as *mut u8, &mut len, std::ptr::null_mut(), 0) == 0 && v == 1 }
}
fn pin_perf() { unsafe { pthread_set_qos_class_self_np(0x21, 0); } }

/// Streaming vector length in bytes. `rdsvl` works outside streaming mode;
/// only call it where `has_sme()`.
unsafe fn svl_bytes() -> u64 { let v: u64; asm!(".arch_extension sme", "rdsvl {0}, #1", out(reg) v, options(nomem, nostack)); v }

/// One 32x32 f32 tile: scratch[32][32] = panel[k][32]^T-ish . b[k][32], i.e.
/// C[i][j] = sum_k panel[k][i] * b[k][j]. Whole tile, always 32 rows; the
/// caller copies the valid ones out. Enters and leaves streaming mode.
#[inline(never)]
unsafe fn sme_tile(panel: *const f32, b: *const f32, k: usize, scratch: *mut f32) {
    asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.s",
        "zero {{ za }}",
        "cbz {k}, 2f",
        "1:",
        "ld1w {{ z0.s }}, p0/z, [{a}]",             // A[0..16][k]   (rows of C)
        "ld1w {{ z1.s }}, p0/z, [{a}, #1, mul vl]",  // A[16..32][k]
        "ld1w {{ z2.s }}, p0/z, [{b}]",             // B[k][0..16]   (columns of C)
        "ld1w {{ z3.s }}, p0/z, [{b}, #1, mul vl]",  // B[k][16..32]
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",      // rows 0..16,  cols 0..16
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",      // rows 0..16,  cols 16..32
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",      // rows 16..32, cols 0..16
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",      // rows 16..32, cols 16..32
        "add {a}, {a}, #128",
        "add {b}, {b}, #128",
        "subs {k}, {k}, #1",
        "b.ne 1b",
        "2:",
        // Row r of the tile is ZA row slice r of (za0, za1) for r < 16 and of
        // (za2, za3) for r >= 16; the slice index is w12 + imm with imm 0..3.
        // ZA stores take a register offset only, hence {c16} = 16 elements.
        "mov {c16}, #16",
        "mov w12, #0",
        "st1w {{ za0h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #4",
        "st1w {{ za0h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #8",
        "st1w {{ za0h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #12",
        "st1w {{ za0h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za0h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za1h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #0",
        "st1w {{ za2h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #4",
        "st1w {{ za2h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #8",
        "st1w {{ za2h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "mov w12, #12",
        "st1w {{ za2h.s[w12, #0] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #0] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #1] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #1] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #2] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #2] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "st1w {{ za2h.s[w12, #3] }}, p0, [{s}]",
        "st1w {{ za3h.s[w12, #3] }}, p0, [{s}, {c16}, lsl #2]",
        "add {s}, {s}, #128",
        "smstop",
        a = inout(reg) panel => _, b = inout(reg) b => _, k = inout(reg) k => _, s = inout(reg) scratch => _,
        c16 = out(reg) _, out("w12") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack),
    );
}

/// 128-byte aligned scratch. SME needs no alignment; this keeps the packed
/// layouts identical to the AMX jig's.
struct Aligned { _keep: Vec<f32>, ptr: *mut f32, len: usize }
impl Aligned {
    fn new(len: usize) -> Self { let mut v = vec![0f32; len + 32]; let p = v.as_mut_ptr(); let off = (128 - (p as usize % 128)) % 128 / 4; Aligned { ptr: unsafe { p.add(off) }, _keep: v, len } }
    fn as_mut(&mut self) -> &mut [f32] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } }
}
unsafe impl Send for Aligned {}

/// A[m x k] (lda) -> ceil(m/32) panels of [k][32], zero-padded, with 4x4 NEON
/// transposes. NEON: must run outside streaming mode, which it does.
unsafe fn pack_a_neon(a: *const f32, lda: usize, m: usize, k: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    for mb in 0..m.div_ceil(32) {
        let panel = out.as_mut_ptr().add(mb * k * 32);
        let ld = |i: usize, kk: usize| -> float32x4_t { let row = mb * 32 + i; if row < m { vld1q_f32(a.add(row * lda + kk)) } else { vdupq_n_f32(0.0) } };
        let mut kk = 0;
        while kk + 4 <= k {
            for i0 in (0..32).step_by(4) {
                let (r0, r1, r2, r3) = (ld(i0, kk), ld(i0 + 1, kk), ld(i0 + 2, kk), ld(i0 + 3, kk));
                let (t0, t1) = (vreinterpretq_f64_f32(vzip1q_f32(r0, r1)), vreinterpretq_f64_f32(vzip2q_f32(r0, r1)));
                let (t2, t3) = (vreinterpretq_f64_f32(vzip1q_f32(r2, r3)), vreinterpretq_f64_f32(vzip2q_f32(r2, r3)));
                vst1q_f32(panel.add(kk * 32 + i0), vreinterpretq_f32_f64(vzip1q_f64(t0, t2)));
                vst1q_f32(panel.add((kk + 1) * 32 + i0), vreinterpretq_f32_f64(vzip2q_f64(t0, t2)));
                vst1q_f32(panel.add((kk + 2) * 32 + i0), vreinterpretq_f32_f64(vzip1q_f64(t1, t3)));
                vst1q_f32(panel.add((kk + 3) * 32 + i0), vreinterpretq_f32_f64(vzip2q_f64(t1, t3)));
            }
            kk += 4;
        }
        while kk < k {
            for i in 0..32 { let row = mb * 32 + i; *panel.add(kk * 32 + i) = if row < m { *a.add(row * lda + kk) } else { 0.0 }; }
            kk += 1;
        }
    }
}

/// B[k x n] -> n/32 blocks of [k][32]: the weights' layout, packed once at load.
fn pack_b(b: &[f32], k: usize, n: usize) -> Aligned {
    let mut out = Aligned::new(k * n);
    let o = out.as_mut();
    for nb in 0..n / 32 { for kk in 0..k { o[nb * k * 32 + kk * 32..][..32].copy_from_slice(&b[kk * n + nb * 32..][..32]); } }
    out
}

/// C[m x n] (ldc) = A[m x k] (lda) . B, with B already `pack_b`'d.
unsafe fn gemm_sme(m: usize, n: usize, k: usize, a: *const f32, lda: usize, bpacked: *const f32, c: *mut f32, ldc: usize, pack: &mut Aligned, scratch: &mut Aligned) {
    assert!(n % 32 == 0, "prototype: n must be a multiple of 32");
    let (mbs, nbs) = (m.div_ceil(32), n / 32);
    assert!(pack.len >= mbs * k * 32);
    pack_a_neon(a, lda, m, k, pack.as_mut());
    for nb in 0..nbs {
        for mb in 0..mbs {
            sme_tile(pack.ptr.add(mb * k * 32), bpacked.add(nb * k * 32), k, scratch.ptr);
            let rows = (m - mb * 32).min(32);
            for r in 0..rows { std::ptr::copy_nonoverlapping(scratch.ptr.add(r * 32), c.add((mb * 32 + r) * ldc + nb * 32), 32); }
        }
    }
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" { fn cblas_sgemm(o: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: c_float, a: *const c_float, lda: c_int, b: *const c_float, ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int); }

fn rnd(n: usize, seed: u32) -> Vec<f32> { let mut s = seed; (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5 }).collect() }

fn concurrent(threads: usize) {
    use std::sync::{Arc, Barrier};
    let shape = std::env::var("SHAPE").unwrap_or("460x512x1536".into());
    let d: Vec<usize> = shape.split('x').map(|v| v.parse().unwrap()).collect();
    let (m, k, n) = (d[0], d[1], d[2]);
    let gf = 2.0 * (m * n * k) as f64 / 1e9; let iters = (2.0e10 / gf / 1e9).clamp(10.0, 200.0) as usize;
    for engine in ["rust-sme", "accelerate"] {
        let barrier = Arc::new(Barrier::new(threads));
        let hs: Vec<_> = (0..threads).map(|t| { let barrier = barrier.clone(); std::thread::spawn(move || {
            pin_perf();
            let (a, b) = (rnd(m * k, 3 + t as u32), rnd(k * n, 4)); let bp = pack_b(&b, k, n);
            let mut c = vec![0f32; m * n]; let mut pack = Aligned::new(m.div_ceil(32) * k * 32); let mut scratch = Aligned::new(32 * 32);
            let mut run = || unsafe { if engine == "rust-sme" { gemm_sme(m, n, k, a.as_ptr(), k, bp.ptr, c.as_mut_ptr(), n, &mut pack, &mut scratch) }
                                       else { cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, c.as_mut_ptr(), n as _) } };
            for _ in 0..3 { run(); }
            barrier.wait(); let t0 = Instant::now(); for _ in 0..iters { run(); } t0.elapsed().as_secs_f64()
        })}).collect();
        let secs: Vec<f64> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let wall = secs.iter().cloned().fold(0.0, f64::max);
        println!("{threads} threads x {m}x{k}x{n}: {engine:10} aggregate {:7.1} GF/s  (per thread {:6.1})", gf * iters as f64 * threads as f64 / wall, gf * iters as f64 / wall);
    }
}

fn main() {
    println!("smeprobe: assembled (smstart / zero za / ld1w / fmopa / st1w za / smstop are in this binary)");
    if !has_sme() { println!("hw.optional.arm.FEAT_SME = 0 on this machine: nothing below has run here."); return; }
    pin_perf();
    let svl = unsafe { svl_bytes() };
    println!("FEAT_SME = 1, SVL = {svl} bytes");
    if svl != 64 { println!("this kernel assumes SVL = 64 bytes (M4); stopping"); return; }
    if let Ok(t) = std::env::var("THREADS") { concurrent(t.parse().unwrap()); return; }
    let mut pack = Aligned::new(2048 * 2048 + 64 * 2048);
    let mut scratch = Aligned::new(32 * 32);
    // --- correctness against Accelerate, awkward sizes first ---
    for &(m, k, n) in &[(32usize, 1usize, 32usize), (32, 4, 32), (45, 37, 64), (33, 5, 32), (460, 512, 1536), (1, 2048, 512)] {
        let (a, b) = (rnd(m * k, 1), rnd(k * n, 2)); let bp = pack_b(&b, k, n);
        let mut c = vec![0f32; m * n]; let mut r = vec![0f32; m * n];
        unsafe { gemm_sme(m, n, k, a.as_ptr(), k, bp.ptr, c.as_mut_ptr(), n, &mut pack, &mut scratch);
                 cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, r.as_mut_ptr(), n as _); }
        let worst = c.iter().zip(&r).map(|(x, y)| (x - y).abs() / (y.abs() + 1e-3)).fold(0f32, f32::max);
        println!("check {m:4}x{k:4}x{n:4}: worst rel diff vs Accelerate {worst:.2e}");
    }
    // --- single-thread throughput on the model's shapes ---
    println!("--- single thread (run with VECLIB_MAXIMUM_THREADS=1) ---");
    for &(m, k, n) in &[(512usize, 512usize, 2048usize), (2048, 512, 1536), (2048, 512, 2048), (2048, 2048, 512), (460, 512, 512)] {
        let (a, b) = (rnd(m * k, 3), rnd(k * n, 4)); let bp = pack_b(&b, k, n);
        let mut c = vec![0f32; m * n];
        let gf = 2.0 * (m * n * k) as f64 / 1e9;
        let time = |f: &mut dyn FnMut()| { for _ in 0..3 { f(); } let t = Instant::now(); let it = 20; for _ in 0..it { f(); } t.elapsed().as_secs_f64() / it as f64 };
        let sme = time(&mut || unsafe { gemm_sme(m, n, k, a.as_ptr(), k, bp.ptr, c.as_mut_ptr(), n, &mut pack, &mut scratch) });
        let acc = time(&mut || unsafe { cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, c.as_mut_ptr(), n as _) });
        println!("{m:4}x{k:4}x{n:4}  rust-sme {:7.1} GF/s   accelerate {:7.1} GF/s   ratio {:.2}", gf / sme, gf / acc, acc / sme);
    }
}

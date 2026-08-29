//! A minimal f32 GEMM on Apple AMX from Rust inline asm, against Accelerate.
//! C[m,n] = A[m,k] @ B[k,n], all row-major. Encodings after corsix/amx.
use std::arch::asm;
use std::os::raw::{c_float, c_int};
use std::time::Instant;

macro_rules! amx {
    ($op:literal, $gpr:expr) => {
        asm!(concat!(".word (0x00201000 + (", stringify!($op), " << 5))"), in("x0") $gpr, options(nostack))
    };
}
unsafe fn amx_set_raw() { asm!("nop", "nop", "nop", ".word 0x00201220", options(nostack)); }
unsafe fn amx_clr_raw() { asm!("nop", "nop", "nop", ".word 0x00201221", options(nostack)); }
thread_local! { static AMX_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
fn set_once() -> bool { std::env::var("AMXSET").map(|v| v == "once").unwrap_or(false) }
/// Enable AMX for this thread: every call in the default mode, once per thread with AMXSET=once.
unsafe fn amx_set() { if set_once() { if !AMX_ON.with(|c| c.replace(true)) { amx_set_raw(); } } else { amx_set_raw(); } }
unsafe fn amx_clr() { if !set_once() { amx_clr_raw(); } }
const INIT_Z: u64 = 1 << 27; // Z = X*Y instead of Z += X*Y (measured: bit 27)
#[inline(always)] unsafe fn stz(p: *mut f32, row: u64) { amx!(5, (p as u64) | (row << 56)) }
#[inline(always)] unsafe fn ldz(p: *const f32, row: u64) { amx!(4, (p as u64) | (row << 56)) }
const fn desc(x: u64, y: u64, z: u64, init: bool) -> u64 { (z << 20) | ((x * 64) << 10) | (y * 64) | if init { INIT_Z } else { 0 } }
/// One K step: X regs (xr, xr+1) <- B row halves, Y regs (yr, yr+1) <- A panel
/// halves, four fma32 into the four Z tiles. Single 64-byte loads — the AMX
/// unit streams those at full rate from L2 — and every descriptor a constant:
/// a K step is ~8 CPU cycles, so nothing may sit between the `.word`s but
/// address arithmetic.
macro_rules! step {
    ($xr:literal, $yr:literal, $init:literal, $b:expr, $a:expr) => {{
        let (b, a): (*const f32, *const f32) = ($b, $a);
        amx!(0, (b as u64) | ($xr << 56)); amx!(0, (b.add(16) as u64) | (($xr + 1) << 56));
        amx!(1, (a as u64) | ($yr << 56)); amx!(1, (a.add(16) as u64) | (($yr + 1) << 56));
        amx!(12, desc($xr, $yr, 0, $init)); amx!(12, desc($xr + 1, $yr, 1, $init));
        amx!(12, desc($xr, $yr + 1, 2, $init)); amx!(12, desc($xr + 1, $yr + 1, 3, $init));
    }};
}
/// Legacy helpers for the unblocked `tile` path.
#[inline(always)] unsafe fn ldx(p: *const f32, reg: u64) { amx!(0, (p as u64) | (reg << 56)); amx!(0, (p.add(16) as u64) | ((reg + 1) << 56)) }
#[inline(always)] unsafe fn ldy(p: *const f32, reg: u64) { amx!(1, (p as u64) | (reg << 56)); amx!(1, (p.add(16) as u64) | ((reg + 1) << 56)) }
#[inline(always)] unsafe fn fma32(x: u64, y: u64, z: u64, init: bool) {
    amx!(12, (z << 20) | ((x * 64) << 10) | (y * 64) | if init { INIT_Z } else { 0 })
}

/// AMX loads and stores need 64/128-byte alignment (a misaligned ldx/ldy is
/// SIGBUS, a misaligned stz silently rounds down). `Vec<f32>` promises 4.
struct Aligned { _keep: Vec<f32>, ptr: *mut f32, len: usize }
impl Aligned {
    fn new(len: usize) -> Self { let mut v = vec![0f32; len + 32]; let p = v.as_mut_ptr(); let off = (128 - (p as usize % 128)) % 128 / 4; Aligned { ptr: unsafe { p.add(off) }, _keep: v, len } }
    fn as_mut(&mut self) -> &mut [f32] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } }
}

/// The same, with 4x4 NEON transposes: four rows at a time, four k at a time.
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

/// A[m x k] (lda) -> ceil(m/32) panels of [k][32], zero-padded.
unsafe fn pack_a(a: *const f32, lda: usize, m: usize, k: usize, out: &mut [f32]) {
    if std::env::var("PACK").map(|v| v == "neon").unwrap_or(true) { return pack_a_neon(a, lda, m, k, out); }
    for mb in 0..m.div_ceil(32) {
        let panel = &mut out[mb * k * 32..][..k * 32];
        for i in 0..32 {
            let row = mb * 32 + i;
            if row < m {
                let src = a.add(row * lda);
                for kk in 0..k { panel[kk * 32 + i] = *src.add(kk); }
            } else {
                for kk in 0..k { panel[kk * 32 + i] = 0.0; }
            }
        }
    }
}

/// A 32x32 tile of an *aligned* C (+)= panel[kl x 32] . b[kl x 32]: Z is
/// loaded from C first when accumulating, and stored back whole. The first K
/// step is peeled so the loop body carries no `init` branch; the body is four
/// K steps on constant registers.
#[inline(never)]
unsafe fn tile_kc(panel: *const f32, b: *const f32, ldb: usize, kl: usize, c: *mut f32, ldc: usize, accumulate: bool) {
    let mut kk = 0;
    if accumulate {
        for t in 0..4u64 { let (xi, yj) = (t & 1, t >> 1); for j in 0..16u64 { ldz(c.add((16 * yj + j) as usize * ldc + 16 * xi as usize), t + 4 * j); } }
    } else {
        step!(0, 0, true, b, panel); kk = 1;
    }
    let (mut bp, mut ap) = (b.add(kk * ldb), panel.add(kk * 32));
    let (bs, as_) = (4 * ldb, 128usize);
    while kk + 4 <= kl {
        step!(0, 0, false, bp, ap);
        step!(2, 2, false, bp.add(ldb), ap.add(32));
        step!(4, 4, false, bp.add(2 * ldb), ap.add(64));
        step!(6, 6, false, bp.add(3 * ldb), ap.add(96));
        bp = bp.add(bs); ap = ap.add(as_); kk += 4;
    }
    while kk < kl { step!(0, 0, false, bp, ap); bp = bp.add(ldb); ap = ap.add(32); kk += 1; }
    for t in 0..4u64 { let (xi, yj) = (t & 1, t >> 1); for j in 0..16u64 { stz(c.add((16 * yj + j) as usize * ldc + 16 * xi as usize), t + 4 * j); } }
}

/// One 32x32 tile of C over the whole K. `rows` valid rows (<= 32).
#[inline(never)]
unsafe fn tile(panel: *const f32, b: *const f32, ldb: usize, k: usize, c: *mut f32, ldc: usize, rows: usize, scratch: *mut f32) {
    let mut kk = 0; let mut first = true;
    // Z[y][x] += X[x] * Y[y]: the row of Z comes from Y, so A (the rows of C)
    // goes to Y and B (the columns) to X, and Z rows come out row-major in C.
    while kk + 4 <= k {
        for u in 0..4 { ldx(b.add((kk + u) * ldb), 2 * u as u64); ldy(panel.add((kk + u) * 32), 2 * u as u64); }
        for u in 0..4 {
            let r = 2 * u as u64;
            fma32(r, r, 0, first); fma32(r + 1, r, 1, first); fma32(r, r + 1, 2, first); fma32(r + 1, r + 1, 3, first);
            first = false;
        }
        kk += 4;
    }
    while kk < k {
        ldx(b.add(kk * ldb), 0); ldy(panel.add(kk * 32), 0);
        fma32(0, 0, 0, first); fma32(1, 0, 1, first); fma32(0, 1, 2, first); fma32(1, 1, 3, first);
        first = false; kk += 1;
    }
    // Tile t = xi + 2*yj holds C rows 16*yj.. (from Y = A) and columns 16*xi..
    // (from X = B); its row j is Z row (t + 4j). Through an aligned tile, then the
    // valid rows to C (whose alignment is the caller's business, not AMX's).
    for t in 0..4u64 {
        let (xi, yj) = (t & 1, t >> 1);
        for j in 0..16u64 { stz(scratch.add((16 * yj + j) as usize * 32 + 16 * xi as usize), t + 4 * j); }
    }
    for r in 0..rows { std::ptr::copy_nonoverlapping(scratch.add(r * 32), c.add(r * ldc), 32); }
}

/// B[k x n] -> n/32 blocks of [k][32], contiguous: the layout the weights take
/// once at load, so every K step reads the next 128 bytes rather than a row
/// 8 KB away.
fn pack_b(b: &[f32], k: usize, n: usize) -> Aligned {
    let mut out = Aligned::new(k * n);
    let o = out.as_mut();
    for nb in 0..n / 32 { for kk in 0..k { o[nb * k * 32 + kk * 32..][..32].copy_from_slice(&b[kk * n + nb * 32..][..32]); } }
    out
}

/// `b` must be 128-byte aligned with `ldb * 4 % 128 == 0` — in the real thing
/// it is the weight matrix, packed once at load. With `bpacked`, `b` is the
/// output of `pack_b` and `ldb` is ignored.
unsafe fn gemm_amx(m: usize, n: usize, k: usize, a: *const f32, lda: usize, b: *const f32, ldb: usize, c: *mut f32, ldc: usize, pack: &mut Aligned, scratch: &mut Aligned, cbuf: &mut Aligned) {
    let bpacked = std::env::var("BPACK").map(|v| v == "1").unwrap_or(false);
    let (bblock, ldb): (Box<dyn Fn(usize) -> *const f32>, usize) = if bpacked { (Box::new(move |nb| b.add(nb * k * 32)), 32) } else { (Box::new(move |nb| b.add(nb * 32)), ldb) };
    assert!(n % 32 == 0, "prototype: n must be a multiple of 32");
    assert!(pack.len >= m.div_ceil(32) * k * 32);
    let t0 = Instant::now();
    pack_a(a, lda, m, k, pack.as_mut());
    PACK_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    amx_set();
    let (mbs, nbs) = (m.div_ceil(32), n / 32);
    let kc: usize = std::env::var("KC").ok().and_then(|v| v.parse().ok()).unwrap_or(k);
    let g: usize = std::env::var("G").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    if kc > 0 {
        // Blocked: G x G tiles share their operands in L1 across a kc block,
        // and partial sums live in an aligned copy of C that Z round-trips to.
        assert!(cbuf.len >= mbs * 32 * n); let cb = cbuf.ptr;
        let mut kc0 = 0;
        while kc0 < k {
            let kl = kc.min(k - kc0);
            for nb0 in (0..nbs).step_by(g) { for mb0 in (0..mbs).step_by(g) {
                for nb in nb0..(nb0 + g).min(nbs) { for mb in mb0..(mb0 + g).min(mbs) {
                    tile_kc(pack.ptr.add(mb * k * 32 + kc0 * 32), bblock(nb).add(kc0 * ldb), ldb, kl, cb.add(mb * 32 * n + nb * 32), n, kc0 > 0);
                } }
            } }
            kc0 += kl;
        }
        amx_clr();
        for r in 0..m { std::ptr::copy_nonoverlapping(cb.add(r * n), c.add(r * ldc), n); }
        return;
    }
    let nb_outer = std::env::var("ORDER").map(|v| v == "nb").unwrap_or(true);
    let do_tile = |mb: usize, nb: usize| {
        let panel = pack.ptr.add(mb * k * 32);
        let rows = (m - mb * 32).min(32);
        tile(panel, bblock(nb), ldb, k, c.add(mb * 32 * ldc + nb * 32), ldc, rows, scratch.ptr);
    };
    if nb_outer { for nb in 0..nbs { for mb in 0..mbs { do_tile(mb, nb); } } }
    else { for mb in 0..mbs { for nb in 0..nbs { do_tile(mb, nb); } } }
    amx_clr();
}
static PACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
unsafe impl Send for Aligned {}

extern "C" { fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32; }
/// QOS_CLASS_USER_INTERACTIVE: keep the measurement on the performance cores,
/// whose AMX unit is the one Accelerate's numbers come from too.
fn pin_perf() { unsafe { pthread_set_qos_class_self_np(0x21, 0); } }

#[link(name = "Accelerate", kind = "framework")]
extern "C" { fn cblas_sgemm(o: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: c_float, a: *const c_float, lda: c_int, b: *const c_float, ldb: c_int, beta: c_float, c: *mut c_float, ldc: c_int); }

fn rnd(n: usize, seed: u32) -> Vec<f32> { let mut s = seed; (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5 }).collect() }

unsafe impl Sync for Aligned {}
fn concurrent(threads: usize) {
    use std::sync::{Arc, Barrier};
    let shape = std::env::var("SHAPE").unwrap_or("460x512x1536".into());
    let d: Vec<usize> = shape.split('x').map(|v| v.parse().unwrap()).collect();
    let (m, k, n) = (d[0], d[1], d[2]);
    let gf = 2.0 * (m * n * k) as f64 / 1e9; let iters = (2.0e10 / gf / 1e9).clamp(10.0, 200.0) as usize;
    for engine in ["rust-amx", "accelerate"] {
        let barrier = Arc::new(Barrier::new(threads));
        let hs: Vec<_> = (0..threads).map(|t| { let barrier = barrier.clone(); std::thread::spawn(move || {
            pin_perf();
            let (a, b) = (rnd(m * k, 3 + t as u32), rnd(k * n, 4));
            let ba = if std::env::var("BPACK").map(|v| v == "1").unwrap_or(false) { pack_b(&b, k, n) } else { aligned_copy(&b) };
            let mut c = vec![0f32; m * n]; let mut pack = Aligned::new(m.div_ceil(32) * k * 32); let mut scratch = Aligned::new(32 * 32); let mut cbuf = Aligned::new(m.div_ceil(32) * 32 * n);
            let mut run = || unsafe { if engine == "rust-amx" { gemm_amx(m, n, k, a.as_ptr(), k, ba.ptr, n, c.as_mut_ptr(), n, &mut pack, &mut scratch, &mut cbuf) }
                                       else { cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, c.as_mut_ptr(), n as _) } };
            for _ in 0..3 { run(); }
            barrier.wait(); let t0 = Instant::now(); for _ in 0..iters { run(); } t0.elapsed().as_secs_f64()
        })}).collect();
        let secs: Vec<f64> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let wall = secs.iter().cloned().fold(0.0, f64::max);
        println!("{threads} threads x {m}x{k}x{n}: {engine:10} aggregate {:7.1} GF/s  (per thread {:6.1})", gf * iters as f64 * threads as f64 / wall, gf * iters as f64 / wall);
    }
}

fn bpacked() -> bool { std::env::var("BPACK").map(|v| v == "1").unwrap_or(false) }
fn b_for_amx(b: &[f32], k: usize, n: usize) -> Aligned { if bpacked() { pack_b(b, k, n) } else { aligned_copy(b) } }

fn aligned_copy(v: &[f32]) -> Aligned { let mut a = Aligned::new(v.len()); a.as_mut().copy_from_slice(v); a }

fn main() {
    pin_perf();
    if let Ok(t) = std::env::var("THREADS") { concurrent(t.parse().unwrap()); return; }
    let mut pack = Aligned::new(2048 * 2048 + 64 * 2048);
    let mut scratch = Aligned::new(32 * 32);
    let mut cbuf = Aligned::new(2048 * 2048 + 32 * 2048);
    // --- minimal cases ---
    for &(m, k, n) in &[(32usize, 1usize, 32usize), (32, 4, 32), (32, 5, 32), (16, 4, 32), (32, 4, 64), (64, 4, 32), (32, 8, 32)] {
        let (a, b) = (rnd(m * k, 1), rnd(k * n, 2)); let ba = b_for_amx(&b, k, n);
        let mut c = vec![0f32; m * n]; let mut r = vec![0f32; m * n];
        unsafe { gemm_amx(m, n, k, a.as_ptr(), k, ba.ptr, n, c.as_mut_ptr(), n, &mut pack, &mut scratch, &mut cbuf);
                 cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, r.as_mut_ptr(), n as _); }
        let worst = c.iter().zip(&r).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let bad = c.iter().zip(&r).filter(|(x, y)| (*x - *y).abs() > 1e-4).count();
        println!("mini {m:3}x{k:2}x{n:3}: worst abs {worst:.3e}, {bad}/{} wrong; c[0..3]={:?} r[0..3]={:?} c[1*n..+3]={:?} r={:?}", m * n, &c[..3], &r[..3], &c[n..n+3], &r[n..n+3]);
    }
    // --- correctness on awkward sizes ---
    for &(m, k, n) in &[(45usize, 37usize, 64usize), (33, 5, 32), (460, 512, 1536), (1, 2048, 512)] {
        let (a, b) = (rnd(m * k, 1), rnd(k * n, 2)); let ba = b_for_amx(&b, k, n);
        let mut c = vec![0f32; m * n]; let mut r = vec![0f32; m * n];
        unsafe { gemm_amx(m, n, k, a.as_ptr(), k, ba.ptr, n, c.as_mut_ptr(), n, &mut pack, &mut scratch, &mut cbuf);
                 cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, r.as_mut_ptr(), n as _); }
        let worst = c.iter().zip(&r).map(|(x, y)| (x - y).abs() / (y.abs() + 1e-3)).fold(0f32, f32::max);
        println!("check {m:4}x{k:4}x{n:4}: worst rel diff vs Accelerate {worst:.2e}");
    }
    // --- single-threaded speed on the model's shapes ---
    println!("--- single thread (run with VECLIB_MAXIMUM_THREADS=1) ---");
    for &(m, k, n) in &[(512usize, 512usize, 2048usize), (2048, 512, 1536), (2048, 512, 2048), (2048, 2048, 512), (460, 512, 512)] {
        let (a, b) = (rnd(m * k, 3), rnd(k * n, 4));
        let ba = if std::env::var("BPACK").map(|v| v == "1").unwrap_or(false) { pack_b(&b, k, n) } else { aligned_copy(&b) };
        let mut c = vec![0f32; m * n];
        let gf = 2.0 * (m * n * k) as f64 / 1e9;
        let time = |f: &mut dyn FnMut()| { for _ in 0..3 { f(); } let t = Instant::now(); let it = 20; for _ in 0..it { f(); } t.elapsed().as_secs_f64() / it as f64 };
        PACK_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        let amx = time(&mut || unsafe { gemm_amx(m, n, k, a.as_ptr(), k, ba.ptr, n, c.as_mut_ptr(), n, &mut pack, &mut scratch, &mut cbuf) });
        let pack_share = PACK_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9 / 23.0 / amx;
        let acc = time(&mut || unsafe { cblas_sgemm(101, 111, 111, m as _, n as _, k as _, 1.0, a.as_ptr(), k as _, b.as_ptr(), n as _, 0.0, c.as_mut_ptr(), n as _) });
        println!("{m:4}x{k:4}x{n:4}  rust-amx {:7.1} GF/s (pack {:2.0}%)   accelerate {:7.1} GF/s   ratio {:.2}", gf / amx, pack_share * 100.0, gf / acc, acc / amx);
    }
}

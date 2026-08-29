use std::arch::asm;
macro_rules! amx { ($op:literal, $gpr:expr) => { asm!(concat!(".word (0x00201000 + (", stringify!($op), " << 5))"), in("x0") $gpr, options(nostack)) }; }
unsafe fn amx_set() { asm!("nop","nop","nop",".word 0x00201220", options(nostack)); }
unsafe fn amx_clr() { asm!("nop","nop","nop",".word 0x00201221", options(nostack)); }
const PAIR: u64 = 1 << 62;

/// 128-byte aligned scratch: returns (vec, aligned ptr)
fn aligned(n: usize) -> (Vec<f32>, *mut f32) { let mut v = vec![0f32; n + 64]; let p = v.as_mut_ptr(); let off = (128 - (p as usize % 128)) % 128 / 4; (v, unsafe { p.add(off) }) }

fn main() { unsafe {
    let (_a, x) = aligned(64); let (_b, y) = aligned(64); let (_c, z) = aligned(64 * 16);
    for i in 0..32 { *x.add(i) = (i + 1) as f32; *y.add(i) = 100.0 * (i + 1) as f32; }
    let mode = std::env::args().nth(1).unwrap_or_default();
    amx_set();
    if mode == "skipbits" {
        // Which bit makes fma32 write X*Y rather than Z += X*Y? Accumulate twice, then once with the bit set.
        for bit in [27u64, 28, 29] {
            amx!(0, x as u64 | PAIR); amx!(1, y as u64 | PAIR);
            amx!(12, 0u64); amx!(12, 0u64);                       // Z tile0 = 2*X*Y
            amx!(12, 1u64 << bit);                                // candidate init
            for r in 0..16u64 { amx!(5, (z.add(r as usize * 16) as u64) | ((4 * r) << 56)); } // tile0 rows
            let got = *z.add(3 * 16 + 5); let want = x.add(5).read() * y.add(3).read();
            println!("bit {bit}: Z[3][5] = {got}  (X*Y = {want}, 3*X*Y = {})  -> {}", 3.0 * want,
                if got == want { "INIT (Z = X*Y)" } else if got == 3.0 * want { "accumulated" } else if got == 2.0*want { "skipped op?" } else { "other" });
            amx_clr(); amx_set();
        }
    } else if mode == "pair" {
        // Does bit 62 load 128 bytes into reg, reg+1? Check X reg 1 / Y reg 1 contents via tiles 1 and 2.
        for bit in [62u64, 61, 60] {
            amx!(0, x as u64 | (1 << bit)); amx!(1, y as u64 | (1 << bit));
            amx!(12, (1u64 << 27) | (1 << 20) | (64 << 10));   // tile1 = X[16..32] x Y[0..16]
            amx!(12, (1u64 << 27) | (2 << 20) | 64);           // tile2 = X[0..16] x Y[16..32]
            for r in 0..16u64 { amx!(5, (z.add(r as usize * 16) as u64) | ((1 + 4 * r) << 56)); }
            let got1 = *z.add(3 * 16 + 5); let want1 = x.add(16 + 5).read() * y.add(3).read();
            for r in 0..16u64 { amx!(5, (z.add(r as usize * 16) as u64) | ((2 + 4 * r) << 56)); }
            let got2 = *z.add(3 * 16 + 5); let want2 = x.add(5).read() * y.add(16 + 3).read();
            println!("bit {bit}: X1 {} ({got1} vs {want1}), Y1 {} ({got2} vs {want2})", if got1 == want1 {"OK"} else {"no"}, if got2 == want2 {"OK"} else {"no"});
            amx_clr(); amx_set();
        }
    } else if mode == "window" {
        // AMX throughput vs working set: both operands stream through a window of `ws` bytes each
        // (2 ldx-pair + 2 ldy-pair... i.e. one 128B X load + one 128B Y load + 4 fma32 per K step).
        use std::time::Instant;
        let big = 8usize << 20; // 8 MB each
        let (_ka, pa) = aligned(big / 4); let (_kb, pb) = aligned(big / 4);
        for i in 0..big / 4 { *pa.add(i) = 1.0; *pb.add(i) = 1.0; }
        println!("{:>10} {:>10}   {:>10}", "A window", "B window", "GFLOP/s");
        let pair = std::env::var("LOADS").map(|v| v != "single").unwrap_or(true);
        println!("loads: {}", if pair { "pair (bit 62)" } else { "two singles" });
        for &(wa, wb) in &[(16usize << 10, 16usize << 10), (32 << 10, 32 << 10), (64 << 10, 64 << 10), (128 << 10, 128 << 10), (256 << 10, 256 << 10), (1 << 20, 1 << 20), (4 << 20, 4 << 20),
                              (16 << 10, 64 << 10), (16 << 10, 1 << 20), (16 << 10, 4 << 20), (64 << 10, 4 << 20)] {
            let steps = 4_000_000u64; let (mut oa, mut ob) = (0usize, 0usize);
            let t = Instant::now();
            for _ in 0..steps / 4 {
                for u in 0..4u64 {
                    let (xb, ya) = (pb.add(ob / 4) as u64, pa.add(oa / 4) as u64);
                    if pair { amx!(0, xb | ((2 * u) << 56) | (1 << 62)); amx!(1, ya | ((2 * u) << 56) | (1 << 62)); }
                    else { amx!(0, xb | ((2 * u) << 56)); amx!(0, (xb + 64) | ((2 * u + 1) << 56)); amx!(1, ya | ((2 * u) << 56)); amx!(1, (ya + 64) | ((2 * u + 1) << 56)); }
                    oa = (oa + 128) & (wa - 1); ob = (ob + 128) & (wb - 1);
                }
                for u in 0..4u64 { let r = 2 * u;
                    amx!(12, (0u64 << 20) | ((r * 64) << 10) | (r * 64)); amx!(12, (1u64 << 20) | (((r + 1) * 64) << 10) | (r * 64));
                    amx!(12, (2u64 << 20) | ((r * 64) << 10) | ((r + 1) * 64)); amx!(12, (3u64 << 20) | (((r + 1) * 64) << 10) | ((r + 1) * 64)); }
            }
            let dt = t.elapsed().as_secs_f64();
            println!("{:>9}K {:>9}K   {:10.1}", wa >> 10, wb >> 10, steps as f64 * 4.0 * 512.0 / dt / 1e9);
        }
    } else if mode.starts_with("align") {
        // Misaligned by 16 bytes: which of ldx / ldy / stz survive?
        let mis = 4usize; // floats
        let which = &mode[5..];
        println!("testing misaligned {which} ...");
        match which {
            "ldx" => { amx!(0, (x.add(mis) as u64) | PAIR); amx!(1, y as u64 | PAIR); }
            "ldy" => { amx!(0, x as u64 | PAIR); amx!(1, (y.add(mis) as u64) | PAIR); }
            _ =>     { amx!(0, x as u64 | PAIR); amx!(1, y as u64 | PAIR); }
        }
        amx!(12, 1u64 << 29);
        let zp = if which == "stz" { z.add(mis) } else { z };
        for r in 0..16u64 { amx!(5, (zp.add(r as usize * 16) as u64) | ((4 * r) << 56)); }
        let (i, j) = (5usize, 3usize);
        let want = match which { "ldx" => x.add(mis + i).read() * y.add(j).read(), "ldy" => x.add(i).read() * y.add(mis + j).read(), _ => x.add(i).read() * y.add(j).read() };
        println!("  {which}: Z[3][5] = {}  want {want}  -> {}", *zp.add(j * 16 + i), if *zp.add(j * 16 + i) == want { "OK" } else { "WRONG" });
    }
    amx_clr();
}}

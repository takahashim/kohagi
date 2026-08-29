# amxprobe

The jig behind the question "can Rust reach Apple's AMX unit the way
Accelerate does?" — an f32 GEMM on M1–M3 AMX from `asm!`, against
`cblas_sgemm`. macOS/aarch64 only; outside the workspace like the other jigs.

- `amxdiag skipbits | pair | window | alignldx | alignldy | alignstz` —
  one measured fact per mode (see below).
- `amxprobe` — correctness against Accelerate (bit-exact), single-thread
  throughput on the model's shapes (`VECLIB_MAXIMUM_THREADS=1`), and with
  `THREADS=n SHAPE=MxKxN` the aggregate of `n` concurrent single-threaded
  callers, which is what Kohagi's fan-out does. Knobs: `BPACK=1` (weights
  pre-packed `[n/32][k][32]`), `G` (tiles sharing operands), `KC` (K block).

## What it established, on an M1 Pro

- Encodings after corsix/amx (`.word 0x00201000 | op << 5`, operand in x0)
  assemble and run under stable rustc. `fma32` matrix mode is
  `Z[y][x] += X[x] * Y[y]`: the Z *row* comes from Y, so A's column panel
  goes to Y and B's row to X for a row-major C. Tile `t = xi + 2*yj` holds
  C rows `16*yj..` and columns `16*xi..`, its row `j` at Z row `t + 4*j`.
- Bit 27 of the `fma32` operand initialises Z (`Z = X*Y`); bits 28/29 do
  something else. Bit 62 of `ldx`/`ldy` loads 128 bytes into two registers.
- Loads and stores need 64-byte alignment: a misaligned `ldx`/`ldy` is
  SIGBUS, a misaligned `stz` silently rounds the address down.
- The unit streams single 64-byte loads at ~1480 GFLOP/s from a working set
  up to 1 MB — i.e. straight from L2, so L1 blocking buys nothing. Pair loads
  collapse to ~340 when the same lines are re-referenced soon.
- A K step is ~8 CPU cycles, so the loop body must be branch-free with
  constant descriptors: the same arithmetic through `OnceLock` reads and an
  `init` branch ran at half speed.
- AMX must be `set` per call: state does not survive an Accelerate call on
  the same thread (its `clr`), and `set` once per thread measured no faster.
- Result: bit-exact with Accelerate; 78–90% of it single-threaded (the rest
  is packing A, which Accelerate overlaps); and with 8 concurrent callers on
  Kohagi's projection shapes, 1.06–1.84x *faster* than Accelerate on five of
  six shapes, since Accelerate's per-call threading does not scale there.

## M4 and later: SME

`smeprobe` is the same GEMM on SME, the documented ARMv9 extension that
replaced AMX on M4: the same packing, the same 32x32 tile through a scratch
buffer, the same driver, the same checks and the same `THREADS=n SHAPE=MxKxN`
concurrency mode — with `smstart`/`smstop`, `ld1w`, `fmopa za0..3.s`,
`st1w {zaNh.s[w12, #i]}` and `zero {za}` in place of the `.word`s. It maps
one for one: the M4's streaming vector length is 512 bits, AMX's 64-byte X/Y,
and ZA is 4 KB with four 32-bit tiles, AMX's Z with its four interleaved
tiles. What differs is in SME's favour: loads need no alignment, partial
tiles could use predicates, `hw.optional.arm.FEAT_SME` is a real sysctl, and
there is `bfmopa` (bf16 in, f32 accumulate) for a `--precision bf16` path
later.

**Written on an M1 and not yet run.** It assembles under stable rustc
(`.arch_extension sme`, all Z and P registers declared clobbered since
`smstart` makes them UNKNOWN), and `main` stops after printing so where
`FEAT_SME` is 0. On an M4:

```console
$ cargo run --release --bin smeprobe                     # correctness, single thread
$ VECLIB_MAXIMUM_THREADS=1 cargo run --release --bin smeprobe
$ THREADS=8 SHAPE=1840x512x1536 cargo run --release --bin smeprobe   # the fan-out
```

The first thing to look at is the `check` lines: with the AMX jig they came
out bit-exact against Accelerate, and that is the bar. Then the two ratios,
against what the M1 gave the AMX jig — 78–90% single-threaded, and 1.06–1.84x
with eight callers on five of six shapes. Whether Accelerate's SME path scales
any better under eight callers than its AMX path did is the open question.

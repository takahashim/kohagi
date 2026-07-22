# Vendored dependencies

## candle-transformers

`candle-transformers` 0.11.0, taken verbatim from crates.io, with the MIT and
Apache-2.0 license texts added from the upstream repository at tag `0.11.0`.

It is vendored so that kohagi can carry patches to candle's ModernBERT that
have not landed upstream yet. Everything outside `src/models/modernbert.rs`
is byte-identical to the published crate, so refreshing to a later release
means re-applying one file's changes rather than reconciling a fork.

The patches themselves are in the commits that follow this one; `git log -p
vendor/candle-transformers/src/models/modernbert.rs` shows what diverged and
why.

## Upstream bugs found while patching

Recorded here rather than filed, for now. Both are in candle 0.11.0.

### Metal `sdpa` mishandles a non-zero start offset

`candle_nn::ops::sdpa` reads q/k/v through their strides correctly, but ignores
`start_offset`. It returns wrong numbers rather than an error, and the error
grows with the offset.

Substituting one tensor at a time against an otherwise contiguous call, with
`b=1, seq=64, hidden=256, heads=4`:

| tensor | start offset | max abs diff |
| --- | ---: | ---: |
| q | 0 | 0 |
| k | h·d | 1.05e-2 |
| v | 2·h·d | 1.48e1 |

Reproduce by slicing a permuted QKV projection — `x.matmul(w).reshape((b, s, 3,
h, d)).permute((2, 0, 3, 1, 4))`, then `narrow(0, i, 1).squeeze(0)` for each of
q/k/v. A `transpose(1, 2)` view of a contiguous tensor is fine, because its
offset is 0; that is a plausible way to conclude the strides are at fault when
they are not.

`call_sdpa_full` is handed a byte offset (`q_l.start_offset() *
size_in_bytes()`), so the fix is likely at the buffer binding rather than in
the shader, which does use `Q_strides`/`K_strides`/`V_strides` as intended.

This is why `ModernBertAttention` splits Wqkv at load: three separate weights
give each projection its own allocation at offset 0, where sdpa is correct.

### `sdpa`'s contiguity precondition is unenforced

`call_sdpa_full` documents "q,k,v are contiguous" and the kernel does assume a
unit stride on the head-dim axis, but nothing validates it. A tensor with a
non-unit last-dim stride — `(b, h, d, s).transpose(2, 3)`, for instance —
silently produces garbage. `metal_fwd` checks dims, dtypes, head-dim membership
and mask shape, so a stride check belongs beside them.

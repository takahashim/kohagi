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

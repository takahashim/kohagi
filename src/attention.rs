//! Which slice of the attention score matrix a computation covers.
//!
//! Both encoders walk the queries in blocks. The f32 path ([`crate::encoder`])
//! does it to keep one score tile inside the memory budget, and both do it to
//! skip the keys a sliding window shuts out. The geometry is the same either
//! way, and its edges are where it goes wrong: the first block, whose window
//! runs off the front of the sequence; the last, which is ragged and whose
//! window runs off the back; and a window already wider than the sequence,
//! where banding computes more than it saves. A mistake in any of them
//! produces embeddings that look reasonable rather than an error, so the
//! arithmetic is written once, here.

/// The queries and keys one call covers, as offsets into a `[seq, seq]` score
/// matrix.
///
/// [`Block::tile`] and [`Block::band`] are the two shapes that exist. The
/// second is the one that drops work, and the reason it may: the keys it
/// leaves out are masked shut, so they contribute `exp(-inf)`, which is an
/// exact zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub(crate) seq: usize,
    pub(crate) q0: usize,
    pub(crate) queries: usize,
    pub(crate) k0: usize,
    pub(crate) keys: usize,
}

impl Block {
    /// The whole matrix, which is what one global layer over a short sequence
    /// wants.
    ///
    /// Only the bf16 encoder asks for it by name; the f32 one arrives at the
    /// same block through [`blocks`].
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub fn full(seq: usize) -> Self {
        Self::tile(seq, 0, seq)
    }

    /// A block of queries against every key: what a global layer needs when
    /// the sequence is too long to score in one piece.
    pub fn tile(seq: usize, q0: usize, queries: usize) -> Self {
        debug_assert!(queries > 0 && q0 + queries <= seq);
        Self {
            seq,
            q0,
            queries,
            k0: 0,
            keys: seq,
        }
    }

    /// The keys a block of queries can reach through a sliding window of
    /// half-width `window`.
    ///
    /// Query `q0 + i` attends `[q0 + i - window, q0 + i + window]`, so the
    /// block as a whole needs the union of those, clamped to the sequence.
    /// Covering every open key is what lets the caller drop the rest of the
    /// row.
    pub fn band(seq: usize, q0: usize, queries: usize, window: usize) -> Self {
        debug_assert!(queries > 0 && q0 + queries <= seq);
        let k0 = q0.saturating_sub(window);
        let last = (q0 + queries - 1 + window).min(seq - 1);
        Self {
            seq,
            q0,
            queries,
            k0,
            keys: last + 1 - k0,
        }
    }

    /// How many queries this block covers, for the caller sizing its buffers.
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Where the block's keys start, and how many there are.
    pub fn keys(&self) -> (usize, usize) {
        (self.k0, self.keys)
    }

    /// Where the block's queries start.
    pub fn q0(&self) -> usize {
        self.q0
    }

    /// Whether this block is the entire score matrix, which is the one case a
    /// caller can compute without narrowing anything or concatenating after.
    pub fn is_full(&self) -> bool {
        self.queries == self.seq && self.keys == self.seq
    }
}

/// The blocks a layer's queries are walked in: `width` at a time, each against
/// the keys `window` leaves open (or every key, when there is no window to
/// narrow by).
pub(crate) fn blocks(
    seq: usize,
    width: usize,
    window: Option<usize>,
) -> impl Iterator<Item = Block> {
    let width = width.clamp(1, seq.max(1));
    (0..seq).step_by(width).map(move |q0| {
        let queries = width.min(seq - q0);
        match window {
            Some(w) => Block::band(seq, q0, queries, w),
            None => Block::tile(seq, q0, queries),
        }
    })
}

/// Whether walking the band beats computing the whole score matrix.
///
/// It cannot help once the window already spans the sequence, and near that
/// point the per-block overhead outweighs what little is masked off, so this
/// wants the band to be a real fraction of the row.
pub(crate) fn banding_pays(seq: usize, window: usize) -> bool {
    seq > 2 * (2 * window + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interior of the band: every key the block's queries can reach, and
    /// none of the rest.
    #[test]
    fn a_band_covers_what_its_queries_reach() {
        let b = Block::band(100, 40, 8, 5);
        // Queries 40..48 reach keys 35..53.
        assert_eq!(b.keys(), (35, 18));
        assert_eq!(b.queries(), 8);
        assert_eq!(b.q0(), 40);
    }

    /// The two edges, where the window runs off the sequence and the clamp is
    /// the only thing keeping the range inside it.
    #[test]
    fn a_band_stops_at_the_ends_of_the_sequence() {
        let first = Block::band(100, 0, 8, 5);
        assert_eq!(first.keys(), (0, 13));
        // The last block is ragged as well as clipped: 4 queries, not 8.
        let last = Block::band(100, 96, 4, 5);
        assert_eq!(last.keys(), (91, 9));
    }

    /// A window wider than the sequence asks for every key, and asks for it
    /// once rather than off the end.
    #[test]
    fn a_window_wider_than_the_sequence_is_the_whole_row() {
        let b = Block::band(10, 0, 10, 40);
        assert_eq!(b.keys(), (0, 10));
        assert!(b.is_full());
        assert!(!banding_pays(10, 40));
        // Which is what makes `banding_pays` an optimization rather than a
        // correctness switch: banding a window this wide would still be right.
        assert_eq!(Block::band(32, 0, 32, 32), Block::full(32));
    }

    /// The band must contain every key the window leaves open for every query
    /// in the block, and nothing is required beyond that. This is the whole
    /// correctness argument for computing a band instead of a full row, so it
    /// is checked exhaustively over the shapes rather than spot-checked.
    #[test]
    fn covers_every_key_the_window_opens() {
        for seq in [1usize, 5, 32, 64, 129, 512] {
            for window in [0usize, 1, 8, 64, 600] {
                for size in [1usize, 3, 32] {
                    for q0 in (0..seq).step_by(size) {
                        let queries = size.min(seq - q0);
                        let b = Block::band(seq, q0, queries, window);
                        let (k0, keys) = b.keys();

                        assert!(
                            k0 + keys <= seq,
                            "seq {seq} w {window} q0 {q0}: past the end"
                        );
                        for i in 0..queries {
                            let q = q0 + i;
                            let lo = q.saturating_sub(window);
                            let hi = (q + window).min(seq - 1);
                            assert!(
                                k0 <= lo && hi < k0 + keys,
                                "seq {seq} w {window} q0 {q0}: query {q} wants {lo}..={hi}, \
                                 block has {k0}..{}",
                                k0 + keys
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every query appears in exactly one block, whatever the width divides
    /// into.
    #[test]
    fn the_blocks_cover_the_queries_once_each() {
        for seq in [1, 7, 8, 9, 64, 100] {
            for width in [1, 3, 8, 64, 1000] {
                let mut next = 0;
                for b in blocks(seq, width, Some(2)) {
                    assert_eq!(b.q0(), next, "seq {seq} width {width}");
                    next += b.queries();
                }
                assert_eq!(next, seq, "seq {seq} width {width}");
            }
        }
    }

    /// Banding is for when the window is a small part of the row. At 512
    /// tokens with ModernBERT's 128-token window it is; at 129 it is not.
    #[test]
    fn banding_pays_only_when_the_window_is_a_fraction_of_the_row() {
        assert!(banding_pays(512, 64));
        assert!(!banding_pays(129, 64));
        assert!(!banding_pays(64, 64));
    }
}

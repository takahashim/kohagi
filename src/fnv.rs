//! FNV-1a, written out rather than taken from `DefaultHasher`.
//!
//! `DefaultHasher`'s algorithm is explicitly allowed to change between Rust
//! releases. Every caller here puts the result in a cache key or a golden test,
//! where a toolchain bump silently invalidating every user's cache — or every
//! recorded hash — is the failure mode.

// Which of these a build uses depends on its features — the compile cache wants
// `write_metadata`, the emitter's golden test wants `hash` — and gating each one
// would say less than this does.
#![allow(dead_code)]

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// An FNV-1a hash being built up from several pieces.
#[derive(Clone, Copy)]
pub struct Hasher(u64);

impl Default for Hasher {
    fn default() -> Self {
        Self(OFFSET)
    }
}

impl Hasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    /// Mix in a file's size and modification time, the stand-in for its contents
    /// when hashing hundreds of megabytes would cost more than it saves.
    pub fn write_metadata(&mut self, meta: &std::fs::Metadata) {
        self.write(&meta.len().to_le_bytes());
        if let Ok(t) = meta.modified() {
            if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                self.write(&d.as_nanos().to_le_bytes());
            }
        }
    }

    pub fn finish(self) -> u64 {
        self.0
    }
}

/// One-shot, for a single slice.
pub fn hash(bytes: &[u8]) -> u64 {
    let mut h = Hasher::new();
    h.write(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_matches_the_published_fnv1a_vectors() {
        // From the FNV reference test vectors, which is what makes this an
        // implementation of a named algorithm rather than whatever it happens to do.
        assert_eq!(hash(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(hash(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn writing_in_pieces_matches_writing_at_once() {
        let mut h = Hasher::new();
        h.write(b"foo");
        h.write(b"bar");
        assert_eq!(h.finish(), hash(b"foobar"));
    }
}

//! The MIL blob storage format that a CoreML model's `weights/weight.bin` uses.
//!
//! Layout, from coremltools `mlmodel/src/MILBlob/Blob/StorageFormat.hpp`
//! (BSD-3-Clause, see `tools/coreml-jigs/LICENSE-COREMLTOOLS-BSD`):
//!
//! ```text
//! |storage_header(64B)|blob_metadata(64B)|data|pad|blob_metadata(64B)|data|...
//! ```
//!
//! Every metadata record sits at the next 64-byte-aligned offset, so the data
//! that follows it is 64-byte aligned too. Entries are a chain rather than a
//! table: the only way to the *n*th record is through the *n-1*th one's size. A
//! MIL program refers to a blob by its **metadata** offset, not its data offset.

/// Byte alignment of every header and metadata record.
pub const ALIGN: u64 = 64;
/// Guard value at the start of each metadata record.
pub const SENTINEL: u32 = 0xDEAD_BEEF;
pub const HEADER_SIZE: u64 = 64;
pub const META_SIZE: u64 = 64;
/// The only `storage_header.version` this format defines (1 was Espresso's).
pub const VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blob {
    /// Offset of the metadata record. This is what MIL's `BLOBFILE(offset:)` uses.
    pub meta_offset: u64,
    pub dtype: u32,
    pub size_in_bytes: u64,
    pub data_offset: u64,
    /// Unused bits in the last byte, for sub-byte dtypes only. Zero otherwise.
    pub padding_bits: u64,
}

impl Blob {
    /// Element count, or `None` for a sub-byte dtype where the size in bytes does
    /// not divide evenly into elements.
    pub fn elements(&self) -> Option<u64> {
        let bits = dtype_bits(self.dtype)?;
        (bits >= 8).then(|| self.size_in_bytes / (bits as u64 / 8))
    }
}

/// `BlobDataType` from coremltools `MILBlob/Blob/BlobDataType.hpp`.
pub fn dtype_name(d: u32) -> &'static str {
    match d {
        1 => "fp16",
        2 => "fp32",
        3 => "uint8",
        4 => "int8",
        5 => "bf16",
        6 => "int16",
        7 => "uint16",
        8 => "int4",
        9 => "uint1",
        10 => "uint2",
        11 => "uint4",
        12 => "uint3",
        13 => "uint6",
        14 => "int32",
        15 => "uint32",
        16 => "fp8e4m3fn",
        17 => "fp8e5m2",
        _ => "unknown",
    }
}

/// Width of a dtype in bits, or `None` if the number is not one we know.
pub fn dtype_bits(d: u32) -> Option<u32> {
    Some(match d {
        1 | 5 | 6 | 7 => 16,
        2 | 14 | 15 => 32,
        3 | 4 | 16 | 17 => 8,
        8 | 11 => 4,
        9 => 1,
        10 => 2,
        12 => 3,
        13 => 6,
        _ => return None,
    })
}

pub fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

fn rd_u32(b: &[u8], off: u64) -> u32 {
    u32::from_le_bytes(b[off as usize..off as usize + 4].try_into().unwrap())
}

fn rd_u64(b: &[u8], off: u64) -> u64 {
    u64::from_le_bytes(b[off as usize..off as usize + 8].try_into().unwrap())
}

/// A structural problem in a blob file. Each variant names the record it was
/// found in, because a `weight.bin` has no other way to be pointed at.
#[derive(Debug)]
pub enum Problem {
    Truncated {
        at: u64,
        need: u64,
        len: u64,
    },
    Version(u32),
    Sentinel {
        index: u32,
        at: u64,
        found: u32,
    },
    Misaligned {
        index: u32,
        at: u64,
    },
    DataOffset {
        index: u32,
        expected: u64,
        found: u64,
    },
    DataPastEof {
        index: u32,
        end: u64,
        len: u64,
    },
    UnknownDtype {
        index: u32,
        dtype: u32,
    },
    StrayPadding {
        index: u32,
        dtype: u32,
        bits: u64,
    },
    TrailingBytes {
        after: u64,
        len: u64,
    },
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { at, need, len } => write!(
                f,
                "file is {len} bytes but a {need}-byte record was expected at {at}"
            ),
            Self::Version(v) => write!(f, "storage_header.version is {v}, expected {VERSION}"),
            Self::Sentinel { index, at, found } => write!(
                f,
                "blob {index}: sentinel at {at} is {found:#x}, expected {SENTINEL:#x} \
                 (the metadata chain is off, so every later offset is suspect)"
            ),
            Self::Misaligned { index, at } => {
                write!(
                    f,
                    "blob {index}: metadata at {at} is not {ALIGN}-byte aligned"
                )
            }
            Self::DataOffset {
                index,
                expected,
                found,
            } => write!(
                f,
                "blob {index}: metadata.offset is {found}, expected {expected} \
                 (data must follow its own metadata immediately)"
            ),
            Self::DataPastEof { index, end, len } => {
                write!(f, "blob {index}: data ends at {end}, past EOF at {len}")
            }
            Self::UnknownDtype { index, dtype } => {
                write!(
                    f,
                    "blob {index}: mil_dtype {dtype} is not a known BlobDataType"
                )
            }
            Self::StrayPadding { index, dtype, bits } => write!(
                f,
                "blob {index}: padding_size_in_bits is {bits} on byte-aligned dtype {}, \
                 expected 0",
                dtype_name(*dtype)
            ),
            Self::TrailingBytes { after, len } => write!(
                f,
                "{} bytes after the last blob (ends at {after}, file is {len})",
                len - after
            ),
        }
    }
}

/// Walk the metadata chain. Returns the declared blob count alongside the blobs
/// actually found, which differ when the chain is broken.
pub fn parse(bytes: &[u8]) -> Result<(u32, Vec<Blob>), Problem> {
    let len = bytes.len() as u64;
    if len < HEADER_SIZE {
        return Err(Problem::Truncated {
            at: 0,
            need: HEADER_SIZE,
            len,
        });
    }
    let count = rd_u32(bytes, 0);
    let version = rd_u32(bytes, 4);
    if version != VERSION {
        return Err(Problem::Version(version));
    }

    let mut blobs = Vec::with_capacity(count as usize);
    let mut pos = HEADER_SIZE;
    for index in 0..count {
        pos = align_up(pos);
        if pos + META_SIZE > len {
            return Err(Problem::Truncated {
                at: pos,
                need: META_SIZE,
                len,
            });
        }
        let found = rd_u32(bytes, pos);
        if found != SENTINEL {
            return Err(Problem::Sentinel {
                index,
                at: pos,
                found,
            });
        }
        let b = Blob {
            meta_offset: pos,
            dtype: rd_u32(bytes, pos + 4),
            size_in_bytes: rd_u64(bytes, pos + 8),
            data_offset: rd_u64(bytes, pos + 16),
            padding_bits: rd_u64(bytes, pos + 24),
        };
        if b.data_offset != pos + META_SIZE {
            return Err(Problem::DataOffset {
                index,
                expected: pos + META_SIZE,
                found: b.data_offset,
            });
        }
        if b.data_offset + b.size_in_bytes > len {
            return Err(Problem::DataPastEof {
                index,
                end: b.data_offset + b.size_in_bytes,
                len,
            });
        }
        pos = b.data_offset + b.size_in_bytes;
        blobs.push(b);
    }
    Ok((count, blobs))
}

/// Everything `parse` accepts but that is still wrong, plus the tail check.
/// Reported as a list rather than the first hit, so one run names every problem.
pub fn lint(bytes: &[u8], blobs: &[Blob]) -> Vec<Problem> {
    let mut out = Vec::new();
    for (i, b) in blobs.iter().enumerate() {
        let index = i as u32;
        if b.meta_offset % ALIGN != 0 {
            out.push(Problem::Misaligned {
                index,
                at: b.meta_offset,
            });
        }
        match dtype_bits(b.dtype) {
            None => out.push(Problem::UnknownDtype {
                index,
                dtype: b.dtype,
            }),
            Some(bits) if bits >= 8 && b.padding_bits != 0 => out.push(Problem::StrayPadding {
                index,
                dtype: b.dtype,
                bits: b.padding_bits,
            }),
            Some(_) => {}
        }
    }
    // The writer appends nothing after the last blob, so trailing bytes mean the
    // file was concatenated or truncated mid-record.
    if let Some(last) = blobs.last() {
        let end = last.data_offset + last.size_in_bytes;
        if (bytes.len() as u64) > align_up(end) {
            out.push(Problem::TrailingBytes {
                after: end,
                len: bytes.len() as u64,
            });
        }
    }
    out
}

/// Re-emit a blob file from parsed records, taking data verbatim from `src`.
/// Byte equality with `src` is what makes this a check on the reader.
pub fn write(blobs: &[Blob], src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; HEADER_SIZE as usize];
    out[0..4].copy_from_slice(&(blobs.len() as u32).to_le_bytes());
    out[4..8].copy_from_slice(&VERSION.to_le_bytes());
    for b in blobs {
        while !(out.len() as u64).is_multiple_of(ALIGN) {
            out.push(0);
        }
        let mut meta = [0u8; META_SIZE as usize];
        meta[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&b.dtype.to_le_bytes());
        meta[8..16].copy_from_slice(&b.size_in_bytes.to_le_bytes());
        meta[16..24].copy_from_slice(&(out.len() as u64 + META_SIZE).to_le_bytes());
        meta[24..32].copy_from_slice(&b.padding_bits.to_le_bytes());
        out.extend_from_slice(&meta);
        let s = b.data_offset as usize;
        out.extend_from_slice(&src[s..s + b.size_in_bytes as usize]);
    }
    out
}

/// The `weights/weight.bin` inside a `.mlmodelc` or `.mlpackage` bundle.
pub fn weight_path(bundle: &std::path::Path) -> Option<std::path::PathBuf> {
    for rel in [
        "weights/weight.bin",
        "Data/com.apple.CoreML/weights/weight.bin",
    ] {
        let p = bundle.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a two-blob file the way the writer would, so the reader has
    /// something to be checked against without a 264MB fixture.
    fn synth() -> Vec<u8> {
        let a = Blob {
            meta_offset: 64,
            dtype: 1,
            size_in_bytes: 8,
            data_offset: 128,
            padding_bits: 0,
        };
        let b = Blob {
            meta_offset: 0,
            dtype: 2,
            size_in_bytes: 16,
            data_offset: 0,
            padding_bits: 0,
        };
        // `write` recomputes offsets, so the placeholder ones above do not matter
        // beyond pointing at data; give it a source long enough to slice.
        let src = vec![0xABu8; 4096];
        write(&[a, b], &src)
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let bytes = synth();
        let (count, blobs) = parse(&bytes).expect("parses");
        assert_eq!(count, 2);
        assert_eq!(blobs.len(), 2);
        assert!(lint(&bytes, &blobs).is_empty());
        assert_eq!(write(&blobs, &bytes), bytes);
    }

    #[test]
    fn offsets_are_aligned_and_chained() {
        let bytes = synth();
        let (_, blobs) = parse(&bytes).unwrap();
        assert_eq!(blobs[0].meta_offset, 64);
        assert_eq!(blobs[0].data_offset, 128);
        // 128 + 8 = 136, which rounds up to 192 for the next metadata record.
        assert_eq!(blobs[1].meta_offset, 192);
        assert_eq!(blobs[1].data_offset, 256);
        for b in &blobs {
            assert_eq!(b.meta_offset % ALIGN, 0);
            assert_eq!(b.data_offset % ALIGN, 0);
        }
    }

    #[test]
    fn a_broken_sentinel_is_named_not_skipped() {
        let mut bytes = synth();
        bytes[192] ^= 0xFF;
        match parse(&bytes) {
            Err(Problem::Sentinel { index, at, .. }) => {
                assert_eq!((index, at), (1, 192));
            }
            other => panic!("expected a sentinel problem, got {other:?}"),
        }
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let mut bytes = synth();
        bytes.extend_from_slice(&[0u8; 128]);
        let (_, blobs) = parse(&bytes).unwrap();
        assert!(matches!(
            lint(&bytes, &blobs).as_slice(),
            [Problem::TrailingBytes { .. }]
        ));
    }

    #[test]
    fn element_counts_follow_dtype_width() {
        let fp16 = Blob {
            meta_offset: 64,
            dtype: 1,
            size_in_bytes: 1024,
            data_offset: 128,
            padding_bits: 0,
        };
        assert_eq!(fp16.elements(), Some(512));
        let int4 = Blob { dtype: 8, ..fp16 };
        assert_eq!(int4.elements(), None);
    }
}

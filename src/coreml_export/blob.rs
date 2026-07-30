//! Writing `weights/weight.bin`, the MIL blob storage file.
//!
//! Format, from coremltools `mlmodel/src/MILBlob/Blob/StorageFormat.hpp`
//! (BSD-3-Clause, see `proto/LICENSE-COREMLTOOLS-BSD`):
//!
//! ```text
//! |storage_header(64B)|blob_metadata(64B)|data|pad|blob_metadata(64B)|data|...
//! ```
//!
//! Each metadata record starts at the next 64-byte boundary, so the data right
//! after it is aligned too. A MIL `const` refers to a blob by its **metadata**
//! offset, which is what [`Writer::write_fp16`] and friends return.
//!
//! The reader half of this lives in `tools/coreml-jigs`, where `milblob` uses it
//! to validate and round-trip a published `weight.bin`; that round-trip is what
//! established the format is written exactly this way.

use half::f16;

const ALIGN: usize = 64;
const SENTINEL: u32 = 0xDEAD_BEEF;
const META_SIZE: usize = 64;
const VERSION: u32 = 2;

/// `BlobDataType` from coremltools `MILBlob/Blob/BlobDataType.hpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlobDataType {
    Float16 = 1,
    Float32 = 2,
    Int8 = 4,
}

/// Accumulates blobs, tracking the offsets a MIL program needs to refer to them.
#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
    count: u32,
}

impl Writer {
    pub fn new() -> Self {
        // The header is rewritten on every append, since it carries the count.
        Self {
            bytes: vec![0u8; ALIGN],
            count: 0,
        }
    }

    /// Append fp16 values, returning the metadata offset to reference them by.
    pub fn write_fp16(&mut self, values: &[f16]) -> u64 {
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        self.append(BlobDataType::Float16, &raw)
    }

    /// Append f32 values as fp16, which is what a converted encoder stores.
    pub fn write_f32_as_fp16(&mut self, values: &[f32]) -> u64 {
        let half: Vec<f16> = values.iter().map(|&v| f16::from_f32(v)).collect();
        self.write_fp16(&half)
    }

    /// Append int8 values, for a weight the graph dequantizes on the way in.
    pub fn write_int8(&mut self, values: &[i8]) -> u64 {
        let raw: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        self.append(BlobDataType::Int8, &raw)
    }

    pub fn write_f32(&mut self, values: &[f32]) -> u64 {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.append(BlobDataType::Float32, &raw)
    }

    fn append(&mut self, dtype: BlobDataType, data: &[u8]) -> u64 {
        while !self.bytes.len().is_multiple_of(ALIGN) {
            self.bytes.push(0);
        }
        let meta_offset = self.bytes.len();
        let data_offset = meta_offset + META_SIZE;

        let mut meta = [0u8; META_SIZE];
        meta[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&(dtype as u32).to_le_bytes());
        meta[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        meta[16..24].copy_from_slice(&(data_offset as u64).to_le_bytes());
        // padding_size_in_bits stays 0: only sub-byte dtypes use it.
        self.bytes.extend_from_slice(&meta);
        self.bytes.extend_from_slice(data);

        self.count += 1;
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        meta_offset as u64
    }

    /// The file, ready to be written to `weights/weight.bin`.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }

    pub fn blob_count(&self) -> u32 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a blob file back the way `milblob` does, so the writer is checked
    /// against the format rather than against itself.
    fn parse(bytes: &[u8]) -> Vec<(u64, u32, u64, u64)> {
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), VERSION);
        let mut out = Vec::new();
        let mut pos = ALIGN;
        for _ in 0..count {
            pos = pos.div_ceil(ALIGN) * ALIGN;
            assert_eq!(
                u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()),
                SENTINEL,
                "bad sentinel at {pos}"
            );
            let dtype = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap());
            let data = u64::from_le_bytes(bytes[pos + 16..pos + 24].try_into().unwrap());
            out.push((pos as u64, dtype, size, data));
            pos = data as usize + size as usize;
        }
        out
    }

    #[test]
    fn offsets_are_aligned_and_data_follows_its_metadata() {
        let mut w = Writer::new();
        let a = w.write_f32_as_fp16(&[1.0, 2.0, 3.0]);
        let b = w.write_f32_as_fp16(&[4.0; 40]);
        let c = w.write_f32(&[0.5, 0.25]);
        assert_eq!(w.blob_count(), 3);
        let bytes = w.finish();

        let blobs = parse(&bytes);
        assert_eq!(blobs.len(), 3);
        assert_eq!([a, b, c], [blobs[0].0, blobs[1].0, blobs[2].0]);
        for (meta, dtype, size, data) in &blobs {
            assert!(
                meta.is_multiple_of(ALIGN as u64),
                "metadata at {meta} is misaligned"
            );
            assert_eq!(*data, meta + META_SIZE as u64);
            assert!(*size > 0);
            assert!(*dtype == 1 || *dtype == 2);
        }
        // fp16 is two bytes per element, f32 four.
        assert_eq!(blobs[0].2, 6);
        assert_eq!(blobs[1].2, 80);
        assert_eq!(blobs[2].2, 8);
        // The first blob sits where a real weight.bin puts it.
        assert_eq!(a, 64);
        assert_eq!(blobs[0].3, 128);
    }

    #[test]
    fn values_survive_the_conversion_to_fp16() {
        let mut w = Writer::new();
        let off = w.write_f32_as_fp16(&[1.0, -2.5, 0.125]);
        let bytes = w.finish();
        let start = off as usize + META_SIZE;
        let read: Vec<f32> = (0..3)
            .map(|i| {
                let o = start + i * 2;
                f16::from_bits(u16::from_le_bytes([bytes[o], bytes[o + 1]])).to_f32()
            })
            .collect();
        // All three are exactly representable in fp16.
        assert_eq!(read, [1.0, -2.5, 0.125]);
    }

    #[test]
    fn an_empty_file_is_still_a_valid_one() {
        let bytes = Writer::new().finish();
        assert_eq!(bytes.len(), ALIGN);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0);
    }
}

//! Regenerate Kohagi's CoreML protobuf bindings from `proto/`.
//!
//!     cargo run --manifest-path tools/coreml-jigs/Cargo.toml --bin gen-proto
//!
//! The output is committed under `src/coreml_proto/generated/`, so building
//! Kohagi needs neither `protoc` nor a build script. This binary is the only
//! thing that needs them, and it only runs when the schema changes.
//!
//! Map fields are emitted as `BTreeMap` rather than `HashMap`: a MIL program
//! carries its functions, block specializations and operation arguments in maps,
//! and serializing a `HashMap` would order them differently on every run. That
//! would make a generated `.mlpackage` unreproducible and the round-trip test
//! meaningless.

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("tools/coreml-jigs sits two levels below the repo root")
        .to_path_buf();
    let proto = root.join("proto");
    let out = root.join("src/coreml_proto/generated");
    std::fs::create_dir_all(&out).expect("create the output directory");

    let mut config = prost_build::Config::new();
    config.out_dir(&out);
    config.btree_map(["."]);
    config
        .compile_protos(
            &[
                proto.join("CoreMLModelSubset.proto"),
                proto.join("MIL.proto"),
            ],
            std::slice::from_ref(&proto),
        )
        .expect("protoc failed; is it installed?");

    println!("wrote {} from {}", out.display(), proto.display());
    for entry in std::fs::read_dir(&out).expect("read the output directory") {
        let path = entry.expect("entry").path();
        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!(
            "  {:>9} bytes  {}",
            len,
            path.file_name().unwrap().to_string_lossy()
        );
    }
}

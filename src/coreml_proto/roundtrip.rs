//! Does the trimmed schema describe a real CoreML model completely?
//!
//! Writing a `.mlpackage` means encoding a [`Model`] that CoreML will accept, and
//! the cheapest way to check an encoder is to hand it a known-good answer:
//! decode a published `model.mlmodel` with these types, encode it again, and
//! compare the bytes. The same reasoning as the `weight.bin` round-trip in
//! [`crate::coreml_export::blob`] — verify the writer while the correct output is
//! still on disk.
//!
//! What that check can and cannot establish, measured against
//! `takahashim/ruri-v3-130m-coreml`:
//!
//! - **The subset is complete.** Re-encoding produced exactly as many bytes as it
//!   consumed, 262,821, and decoding the result gave back an equal [`Model`]. A
//!   field with nowhere to go would have been dropped — prost does not preserve
//!   unknown fields — and the output would have been shorter.
//! - **Byte-identity is not reachable, and not a defect.** The first difference
//!   is at byte 95, the first key of `Metadata.userDefined`. protobuf does not
//!   define an order for map fields: coremltools emits Python's insertion order,
//!   these bindings use `BTreeMap` and emit sorted order. Sorted is the better
//!   choice here, because package generation has to be reproducible, and a
//!   `HashMap` would reorder on every run.
//!
//! So the assertion is losslessness rather than byte equality, and the byte
//! comparison is reported instead. A shorter re-encoding is the failure that
//! matters, and it is asserted directly.
//!
//! The fixture is not in the repository: it is 262KB of someone else's model, and
//! the weights beside it are 264MB. Point the test at one with
//! `KOHAGI_TEST_MLMODEL`, or it skips.

use prost::Message;

use super::Model;

fn fixture() -> Option<Vec<u8>> {
    let path = std::env::var_os("KOHAGI_TEST_MLMODEL")?;
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(e) => panic!(
            "KOHAGI_TEST_MLMODEL points at {}, which could not be read: {e}",
            std::path::Path::new(&path).display()
        ),
    }
}

#[test]
fn a_published_model_survives_decode_and_encode() {
    let Some(original) = fixture() else {
        eprintln!("skipping: set KOHAGI_TEST_MLMODEL to a model.mlmodel to run this");
        return;
    };

    let model = Model::decode(original.as_slice()).expect("decodes with the subset schema");
    let again = model.encode_to_vec();

    // Nothing was dropped: prost silently discards fields the schema cannot name,
    // and a discarded field makes the output shorter. Length equality is the
    // check that `proto/CoreMLModelSubset.proto` covers this file.
    assert_eq!(
        again.len(),
        original.len(),
        "re-encoding changed the size, so a field in this model has nowhere to be \
         decoded into; add it to proto/CoreMLModelSubset.proto"
    );

    // And nothing was altered: decoding our own output must give the same model.
    let decoded_again = Model::decode(again.as_slice()).expect("our own output decodes back");
    assert_eq!(
        decoded_again, model,
        "re-encoding changed the model's contents"
    );

    if again != original {
        let first = again
            .iter()
            .zip(&original)
            .position(|(a, b)| a != b)
            .expect("the lengths are equal, so a difference is inside both");
        eprintln!(
            "note: {} bytes, lossless, but not byte-identical — first difference at \
             {first}. Expected: protobuf leaves map ordering undefined, and these \
             bindings sort where coremltools does not.",
            original.len()
        );
    }
}

#[test]
fn the_program_is_reachable_and_shaped_as_expected() {
    let Some(bytes) = fixture() else {
        eprintln!("skipping: set KOHAGI_TEST_MLMODEL to a model.mlmodel to run this");
        return;
    };
    let model = Model::decode(bytes.as_slice()).expect("decodes");

    // The three fields a converted encoder uses, reached through the generated
    // types rather than by walking the wire — this is what an emitter will build.
    assert!(model.specification_version > 0);
    let description = model.description.as_ref().expect("a model description");
    let names: Vec<&str> = description.input.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"input_ids") && names.contains(&"attention_mask"),
        "inputs were {names:?}"
    );
    assert!(description.output.iter().any(|f| f.name == "hidden"));

    let Some(super::model::Type::MlProgram(program)) = &model.r#type else {
        panic!("not an ML Program");
    };
    let main = program
        .functions
        .get("main")
        .expect("a function named main");
    let block = main
        .block_specializations
        .values()
        .next()
        .expect("a block specialization");

    // 1539 operations for ruri-v3-130m at seq 128. Assert the shape of the graph
    // rather than the number, so the test holds for any converted encoder.
    assert!(
        block.operations.len() > 100,
        "only {} operations",
        block.operations.len()
    );
    let consts = block
        .operations
        .iter()
        .filter(|op| op.r#type == "const")
        .count();
    assert!(consts > 0, "no const operations, so no weights are bound");
    assert!(
        block.operations.iter().any(|op| op.r#type == "linear"),
        "no linear operations in what should be a transformer"
    );
}

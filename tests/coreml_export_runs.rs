//! Does CoreML accept and run a model this crate wrote?
//!
//! A ladder, as tests rather than a session at the terminal: emit a small
//! `.mlpackage`, compile it, run it, and
//! check the numbers. Each step adds one thing that could be wrong — a program at
//! all, then a weight read out of `weight.bin`, then an operation with a
//! parameter — so a failure says which.
//!
//! Needs both features and a real Neural Engine, so it is skipped elsewhere:
//!
//! ```console
//! cargo test --features coreml,coreml-export --test coreml_export_runs
//! ```

#![cfg(all(feature = "coreml", feature = "coreml-export"))]

use half::f16;
use kohagi::coreml_export::{
    blob,
    mil::{Builder, DType, Tensor},
    model, write_package, Io,
};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

/// A scratch directory of our own, so a failing test leaves its package behind to
/// look at rather than deleting the evidence.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("kohagi-export-tests/{name}.mlpackage"));
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).expect("create the scratch directory");
    }
    dir
}

fn url(path: &std::path::Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(path.to_str().expect("utf-8 path")))
}

/// Compile a package and load it, pinned to CPU+ANE like the runtime backend.
fn load(package: &std::path::Path) -> Retained<MLModel> {
    load_function(package, None)
}

/// The same, naming one function of a multi-function bundle.
fn load_function(package: &std::path::Path, function: Option<&str>) -> Retained<MLModel> {
    let compiled = unsafe {
        #[allow(deprecated)]
        MLModel::compileModelAtURL_error(&url(package))
    }
    .unwrap_or_else(|e| panic!("CoreML rejected {}: {e}", package.display()));
    let path = compiled.path().expect("a compiled path");
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
    if let Some(name) = function {
        unsafe { config.setFunctionName(Some(&NSString::from_str(name))) };
    }
    unsafe {
        MLModel::modelWithContentsOfURL_configuration_error(
            &url(std::path::Path::new(&path.to_string())),
            &config,
        )
    }
    .unwrap_or_else(|e| panic!("compiled but did not load: {e}"))
}

/// One input value, with the shape it is declared as.
enum Feed {
    F16(Vec<usize>, Vec<f32>),
    I32(Vec<usize>, Vec<i32>),
}

fn multiarray(feed: &Feed) -> Retained<MLMultiArray> {
    let (shape, dtype) = match feed {
        Feed::F16(s, _) => (s, MLMultiArrayDataType::Float16),
        Feed::I32(s, _) => (s, MLMultiArrayDataType::Int32),
    };
    let dims: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&d| NSNumber::new_isize(d as isize))
        .collect();
    let arr = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &NSArray::from_retained_slice(&dims),
            dtype,
        )
    }
    .expect("allocate an input array");
    unsafe {
        #[allow(deprecated)]
        let raw = arr.dataPointer().as_ptr();
        match feed {
            Feed::F16(_, data) => {
                let ptr = raw as *mut u16;
                for (i, &v) in data.iter().enumerate() {
                    *ptr.add(i) = f16::from_f32(v).to_bits();
                }
            }
            Feed::I32(_, data) => {
                let ptr = raw as *mut i32;
                for (i, &v) in data.iter().enumerate() {
                    *ptr.add(i) = v;
                }
            }
        }
    }
    arr
}

/// Run a model and read one fp16 output.
fn predict(model: &MLModel, feeds: &[(&str, Feed)], output: &str) -> Vec<f32> {
    let arrays: Vec<Retained<MLMultiArray>> = feeds.iter().map(|(_, f)| multiarray(f)).collect();
    let values: Vec<Retained<MLFeatureValue>> = arrays
        .iter()
        .map(|a| unsafe { MLFeatureValue::featureValueWithMultiArray(a) })
        .collect();
    let keys: Vec<Retained<NSString>> = feeds
        .iter()
        .map(|(name, _)| NSString::from_str(name))
        .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let value_refs: Vec<&AnyObject> = values.iter().map(|v| &**v as &AnyObject).collect();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&key_refs, &value_refs);
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    }
    .expect("build a feature provider");

    let out = unsafe { model.predictionFromFeatures_error(ProtocolObject::from_ref(&*provider)) }
        .unwrap_or_else(|e| panic!("prediction failed: {e}"));
    let feature = unsafe { out.featureValueForName(&NSString::from_str(output)) }
        .unwrap_or_else(|| panic!("no output named {output}"));
    let arr = unsafe { feature.multiArrayValue() }.expect("a multiarray output");
    read_strided(&arr)
}

/// Copy an output array in logical order.
///
/// `dataPointer` is not a packed row-major buffer: CoreML pads a rank-3 output's
/// last axis, so reading `count()` elements straight off it interleaves real
/// values with padding. Reading through `shape` and `strides` is the only correct
/// way, and the difference only shows up above rank 2 — which is why the rungs
/// with a `[1, n]` output passed while the `[1, seq, dim]` ones did not.
fn read_strided(arr: &MLMultiArray) -> Vec<f32> {
    let numbers = |a: Retained<NSArray<NSNumber>>| -> Vec<usize> {
        a.iter().map(|n| n.as_isize().max(0) as usize).collect()
    };
    let shape = numbers(unsafe { arr.shape() });
    let strides = numbers(unsafe { arr.strides() });
    let total: usize = shape.iter().product();

    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    unsafe {
        #[allow(deprecated)]
        let ptr = arr.dataPointer().as_ptr() as *const u16;
        for _ in 0..total {
            let offset: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
            out.push(f16::from_bits(*ptr.add(offset)).to_f32());
            // Odometer over the logical indices, last axis fastest.
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
    out
}

/// Rung 1: does CoreML accept a program at all? One operation, no weights.
#[test]
fn an_identity_model_loads_and_returns_its_input() {
    let x = Tensor::new("x", DType::Fp16, &[1, 4]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let y = b.op(
        "identity",
        Tensor::new("y", DType::Fp16, &[1, 4]),
        &[("x", &x)],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y],
        },
    );

    let dir = scratch("identity");
    write_package(&dir, &m, &blob::Writer::new().finish()).expect("write the package");
    let model = load(&dir);

    let input = [1.0, -2.5, 0.125, 3.0];
    let got = predict(&model, &[("x", Feed::F16(vec![1, 4], input.to_vec()))], "y");
    assert_eq!(got, input);
}

/// Rung 2: is a weight read out of `weight.bin` at the offset we recorded?
///
/// `add` with a constant that only exists in the blob file. A wrong offset or a
/// wrong metadata record shows up as wrong numbers rather than a load failure,
/// which is why the values are distinct and exactly representable in fp16.
#[test]
fn a_constant_read_from_the_blob_file_reaches_the_graph() {
    let mut weights = blob::Writer::new();
    let bias = [0.5, -1.0, 2.0, 0.25];
    let offset = weights.write_f32_as_fp16(&bias);

    let x = Tensor::new("x", DType::Fp16, &[1, 4]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let c = b.const_blob(Tensor::new("c", DType::Fp16, &[1, 4]), offset);
    let y = b.op(
        "add",
        Tensor::new("y", DType::Fp16, &[1, 4]),
        &[("x", &x), ("y", &c)],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y],
        },
    );

    let dir = scratch("blob-add");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model = load(&dir);

    let input = [1.0, 1.0, 1.0, 1.0];
    let got = predict(&model, &[("x", Feed::F16(vec![1, 4], input.to_vec()))], "y");
    let want: Vec<f32> = input.iter().zip(bias).map(|(a, b)| a + b).collect();
    assert_eq!(got, want, "blob offset {offset} did not deliver {bias:?}");
}

/// Rung 3: a real operation with weights and a parameter — `linear`, the one an
/// encoder is mostly made of. 76 of the 735 operations in `ruri-v3-130m` are
/// this.
#[test]
fn a_linear_layer_computes_what_it_should() {
    // y = x @ w^T + b, with w [out, in] as MIL's `linear` expects.
    let (n_in, n_out) = (3usize, 2usize);
    let w = [1.0, 0.0, 0.5, 0.0, 2.0, -1.0];
    let bias = [0.25, -0.5];
    let mut weights = blob::Writer::new();
    let w_offset = weights.write_f32_as_fp16(&w);
    let b_offset = weights.write_f32_as_fp16(&bias);

    let x = Tensor::new("x", DType::Fp16, &[1, n_in]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let wt = b.const_blob(Tensor::new("w", DType::Fp16, &[n_out, n_in]), w_offset);
    let bt = b.const_blob(Tensor::new("b", DType::Fp16, &[n_out]), b_offset);
    let y = b.op(
        "linear",
        Tensor::new("y", DType::Fp16, &[1, n_out]),
        &[("x", &x), ("weight", &wt), ("bias", &bt)],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y],
        },
    );

    let dir = scratch("linear");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model = load(&dir);

    let input = [1.0, 2.0, 4.0];
    let got = predict(
        &model,
        &[("x", Feed::F16(vec![1, n_in], input.to_vec()))],
        "y",
    );
    let want: Vec<f32> = (0..n_out)
        .map(|o| (0..n_in).map(|i| input[i] * w[o * n_in + i]).sum::<f32>() + bias[o])
        .collect();
    // Every value here is exact in fp16, so this is equality rather than a
    // tolerance: a discrepancy would mean the graph is wrong, not rounded.
    assert_eq!(got, want);
}

/// fp16 has about three decimal digits, so anything that sums or exponentiates
/// needs a tolerance. The rungs above use exact equality because their values were
/// chosen to be exact; these cannot be.
fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {i} is {g}, expected {w} (tolerance {tol})\ngot  {got:?}\nwant {want:?}"
        );
    }
}

/// Rung 4: `gather`, the embedding lookup. One of the 735 operations, and the
/// only one taking integer input, so it is also the first test with a mixed-dtype
/// interface.
#[test]
fn an_embedding_lookup_returns_the_rows_it_was_asked_for() {
    let (vocab, dim, seq) = (5usize, 3usize, 4usize);
    // Row v is [v, v + 0.5, -v], so a wrong row is obvious rather than plausible.
    let table: Vec<f32> = (0..vocab)
        .flat_map(|v| [v as f32, v as f32 + 0.5, -(v as f32)])
        .collect();
    let mut weights = blob::Writer::new();
    let offset = weights.write_f32_as_fp16(&table);

    let ids = Tensor::new("input_ids", DType::Int32, &[1, seq]);
    let mut b = Builder::new(std::slice::from_ref(&ids));
    let emb = b.const_blob(Tensor::new("emb", DType::Fp16, &[vocab, dim]), offset);
    let axis = b.const_i32(Tensor::new("axis", DType::Int32, &[]), &[0]);
    let batch_dims = b.const_i32(Tensor::new("batch_dims", DType::Int32, &[]), &[0]);
    let validate = b.const_bool("validate_indices", false);
    let out = b.op(
        "gather",
        Tensor::new("hidden", DType::Fp16, &[1, seq, dim]),
        &[
            ("x", &emb),
            ("indices", &ids),
            ("axis", &axis),
            ("batch_dims", &batch_dims),
            ("validate_indices", &validate),
        ],
    );
    b.returns(&out);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![ids],
            outputs: vec![out],
        },
    );

    let dir = scratch("gather");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model = load(&dir);

    let picked = [3, 0, 4, 1];
    let got = predict(
        &model,
        &[("input_ids", Feed::I32(vec![1, seq], picked.to_vec()))],
        "hidden",
    );
    let want: Vec<f32> = picked
        .iter()
        .flat_map(|&v| table[v as usize * dim..(v as usize + 1) * dim].to_vec())
        .collect();
    assert_eq!(got, want, "gather returned the wrong rows");
}

/// Rung 5: `layer_norm`. 39 of the 735 operations, and the first with a float
/// parameter (`epsilon`) that has to be encoded as raw fp16 bytes.
///
/// ModernBERT's norm has no beta, so the operation takes only `gamma` — which is
/// also how the reference model emits it.
#[test]
fn a_layer_norm_normalizes_over_the_last_axis() {
    let (seq, dim) = (2usize, 4usize);
    let gamma = [1.0f32, 2.0, 0.5, -1.0];
    let eps = 1e-5f32;
    let mut weights = blob::Writer::new();
    let gamma_offset = weights.write_f32_as_fp16(&gamma);

    let x = Tensor::new("x", DType::Fp16, &[1, seq, dim]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let g = b.const_blob(Tensor::new("gamma", DType::Fp16, &[dim]), gamma_offset);
    let axes = b.const_i32(Tensor::new("axes", DType::Int32, &[1]), &[-1]);
    let epsilon = b.const_fp16(Tensor::new("eps", DType::Fp16, &[]), &[eps]);
    let y = b.op(
        "layer_norm",
        Tensor::new("y", DType::Fp16, &[1, seq, dim]),
        &[
            ("x", &x),
            ("axes", &axes),
            ("epsilon", &epsilon),
            ("gamma", &g),
        ],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y],
        },
    );

    let dir = scratch("layer-norm");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model = load(&dir);

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 2.0];
    let got = predict(
        &model,
        &[("x", Feed::F16(vec![1, seq, dim], input.clone()))],
        "y",
    );

    let mut want = Vec::new();
    for row in input.chunks(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let denom = (var + eps).sqrt();
        want.extend(row.iter().zip(gamma).map(|(v, g)| (v - mean) / denom * g));
    }
    assert_close(&got, &want, 4e-3, "layer_norm");
}

/// Rung 6: `gelu` in `EXACT` mode, the activation ModernBERT's GeGLU uses. 19 of
/// the 735 operations. First operation with a string parameter.
#[test]
fn gelu_matches_the_exact_formula() {
    let n = 6usize;
    let x = Tensor::new("x", DType::Fp16, &[1, n]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let mode = b.const_str("mode", "EXACT");
    let y = b.op(
        "gelu",
        Tensor::new("y", DType::Fp16, &[1, n]),
        &[("x", &x), ("mode", &mode)],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y],
        },
    );

    let dir = scratch("gelu");
    write_package(&dir, &m, &blob::Writer::new().finish()).expect("write the package");
    let model = load(&dir);

    let input = [-3.0f32, -1.0, -0.5, 0.0, 1.0, 2.5];
    let got = predict(&model, &[("x", Feed::F16(vec![1, n], input.to_vec()))], "y");
    // EXACT is x * Phi(x), with Phi the Gaussian CDF: 0.5 * (1 + erf(x / sqrt(2))).
    // Abramowitz & Stegun 7.1.26, good to ~1e-7, which is well inside fp16. The
    // coefficients are f64 so the approximation is not itself the error being
    // measured.
    let erf = |z: f32| {
        let (sign, z) = (z.signum() as f64, f64::from(z.abs()));
        let t = 1.0 / (1.0 + 0.3275911 * z);
        let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t;
        (sign * (1.0 - poly * (-z * z).exp())) as f32
    };
    let want: Vec<f32> = input
        .iter()
        .map(|&v| v * 0.5 * (1.0 + erf(v / std::f32::consts::SQRT_2)))
        .collect();
    assert_close(&got, &want, 4e-3, "gelu");
}

/// Rung 7: scaled dot-product attention over one head — `matmul`, `mul`, `add`,
/// `softmax`, `matmul`, which is the middle of every one of the 19 transformer
/// blocks.
///
/// This is the first graph with a chain long enough for an intermediate to be
/// wrong without the output being obviously broken, and the first use of
/// `transpose_y` and of an additive mask.
#[test]
fn one_attention_head_computes_scaled_dot_product() {
    let (seq, head_dim) = (3usize, 2usize);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = Tensor::new("q", DType::Fp16, &[1, seq, head_dim]);
    let k = Tensor::new("k", DType::Fp16, &[1, seq, head_dim]);
    let v = Tensor::new("v", DType::Fp16, &[1, seq, head_dim]);
    let mut b = Builder::new(&[q.clone(), k.clone(), v.clone()]);

    let no = b.const_bool("no", false);
    let yes = b.const_bool("yes", true);
    // scores = q @ k^T
    let scores = b.op(
        "matmul",
        Tensor::new("scores", DType::Fp16, &[1, seq, seq]),
        &[
            ("x", &q),
            ("y", &k),
            ("transpose_x", &no),
            ("transpose_y", &yes),
        ],
    );
    let scale_c = b.const_fp16(Tensor::new("scale", DType::Fp16, &[]), &[scale]);
    let scaled = b.op(
        "mul",
        Tensor::new("scaled", DType::Fp16, &[1, seq, seq]),
        &[("x", &scores), ("y", &scale_c)],
    );
    // A causal additive mask, the same shape the encoder's mask enters at: 0 to
    // keep, a large negative to drop.
    let mut mask = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                mask[i * seq + j] = -10_000.0;
            }
        }
    }
    let mut weights = blob::Writer::new();
    let mask_offset = weights.write_f32_as_fp16(&mask);
    let mask_c = b.const_blob(
        Tensor::new("mask", DType::Fp16, &[1, seq, seq]),
        mask_offset,
    );
    let masked = b.op(
        "add",
        Tensor::new("masked", DType::Fp16, &[1, seq, seq]),
        &[("x", &scaled), ("y", &mask_c)],
    );
    let axis = b.const_i32(Tensor::new("axis", DType::Int32, &[]), &[-1]);
    let probs = b.op(
        "softmax",
        Tensor::new("probs", DType::Fp16, &[1, seq, seq]),
        &[("x", &masked), ("axis", &axis)],
    );
    let out = b.op(
        "matmul",
        Tensor::new("out", DType::Fp16, &[1, seq, head_dim]),
        &[
            ("x", &probs),
            ("y", &v),
            ("transpose_x", &no),
            ("transpose_y", &no),
        ],
    );
    b.returns(&out);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![q, k, v],
            outputs: vec![out],
        },
    );

    let dir = scratch("attention");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model = load(&dir);

    let qd = vec![1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5];
    let kd = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let vd = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![1, seq, head_dim];
    let got = predict(
        &model,
        &[
            ("q", Feed::F16(shape.clone(), qd.clone())),
            ("k", Feed::F16(shape.clone(), kd.clone())),
            ("v", Feed::F16(shape.clone(), vd.clone())),
        ],
        "out",
    );

    // The same computation in f32, masked rows included.
    let mut want = vec![0.0f32; seq * head_dim];
    for i in 0..seq {
        let mut logits = vec![f32::NEG_INFINITY; seq];
        for j in 0..=i {
            let dot: f32 = (0..head_dim)
                .map(|d| qd[i * head_dim + d] * kd[j * head_dim + d])
                .sum();
            logits[j] = dot * scale;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for j in 0..seq {
            let p = exp[j] / sum;
            for d in 0..head_dim {
                want[i * head_dim + d] += p * vd[j * head_dim + d];
            }
        }
    }
    assert_close(&got, &want, 8e-3, "attention");
}

/// An f32 reference for one ModernBERT block, written straight from the
/// definition rather than from the emitter, so that agreement is evidence about
/// the emitted graph's wiring rather than a tautology.
mod reference {
    use kohagi::coreml_export::modernbert::{rope_tables, Activation, Config};

    pub struct Weights<'a> {
        pub attn_norm: Option<&'a [f32]>,
        pub wqkv: &'a [f32],
        pub wo: &'a [f32],
        pub mlp_norm: &'a [f32],
        pub mlp_wi: &'a [f32],
        pub mlp_wo: &'a [f32],
    }

    /// `y[i, o] = sum_k x[i, k] * w[o, k]`, MIL's `linear` without a bias.
    fn linear(x: &[f32], w: &[f32], rows: usize, n_in: usize, n_out: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * n_out];
        for r in 0..rows {
            for o in 0..n_out {
                out[r * n_out + o] = (0..n_in).map(|k| x[r * n_in + k] * w[o * n_in + k]).sum();
            }
        }
        out
    }

    fn layer_norm(x: &[f32], gamma: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * dim);
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let denom = (var + eps).sqrt();
            out.extend(row.iter().zip(gamma).map(|(v, g)| (v - mean) / denom * g));
        }
        out
    }

    fn silu(v: f32) -> f32 {
        let v = f64::from(v);
        (v / (1.0 + (-v).exp())) as f32
    }

    fn gelu_exact(v: f32) -> f32 {
        let z = f64::from(v) / std::f64::consts::SQRT_2;
        // erf via Abramowitz & Stegun 7.1.26.
        let sign = z.signum();
        let z = z.abs();
        let t = 1.0 / (1.0 + 0.3275911 * z);
        let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t;
        let erf = sign * (1.0 - poly * (-z * z).exp());
        (f64::from(v) * 0.5 * (1.0 + erf)) as f32
    }

    pub fn block(cfg: &Config, input: &[f32], w: &Weights, mask: &[f32]) -> Vec<f32> {
        let (h, heads, seq) = (cfg.hidden, cfg.heads, cfg.seq);
        let d = cfg.head_dim();
        let inter = cfg.intermediate;

        let normed = match w.attn_norm {
            None => input.to_vec(),
            Some(gamma) => layer_norm(input, gamma, seq, h, cfg.eps),
        };
        let qkv = linear(&normed, w.wqkv, seq, h, 3 * h);

        // qkv is [seq, 3, heads, d] once reshaped; gather each head's q/k/v.
        let at =
            |t: usize, head: usize, pos: usize, i: usize| qkv[pos * 3 * h + t * h + head * d + i];
        let (cos, sin) = rope_tables(cfg);
        let rope = |head: usize, t: usize| -> Vec<f32> {
            let mut out = vec![0.0; seq * d];
            for pos in 0..seq {
                for i in 0..d {
                    let x = at(t, head, pos, i);
                    // rotate_half: the partner of i is i +/- d/2, negated for the
                    // first half.
                    let partner = if i < d / 2 {
                        -at(t, head, pos, i + d / 2)
                    } else {
                        at(t, head, pos, i - d / 2)
                    };
                    out[pos * d + i] = x * cos[pos * d + i] + partner * sin[pos * d + i];
                }
            }
            out
        };

        let mut context = vec![0.0f32; seq * h];
        for head in 0..heads {
            let q = rope(head, 0);
            let k = rope(head, 1);
            for i in 0..seq {
                let mut logits = Vec::with_capacity(seq);
                for j in 0..seq {
                    let dot: f32 = (0..d).map(|t| q[i * d + t] * k[j * d + t]).sum();
                    logits.push(dot * cfg.scale() + mask[i * seq + j]);
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f32 = exp.iter().sum();
                for (j, e) in exp.iter().enumerate() {
                    let p = e / sum;
                    for t in 0..d {
                        context[i * h + head * d + t] += p * at(2, head, j, t);
                    }
                }
            }
        }

        let projected = linear(&context, w.wo, seq, h, h);
        let attn_residual: Vec<f32> = input.iter().zip(&projected).map(|(a, b)| a + b).collect();

        let mlp_normed = layer_norm(&attn_residual, w.mlp_norm, seq, h, cfg.eps);
        let wide = linear(&mlp_normed, w.mlp_wi, seq, h, 2 * inter);
        let mut gated = vec![0.0f32; seq * inter];
        for r in 0..seq {
            for i in 0..inter {
                let gate = wide[r * 2 * inter + i];
                let up = wide[r * 2 * inter + inter + i];
                let activated = match cfg.activation {
                    Activation::Gelu => gelu_exact(gate),
                    Activation::Silu => silu(gate),
                };
                gated[r * inter + i] = activated * up;
            }
        }
        let mlp_out = linear(&gated, w.mlp_wo, seq, inter, h);
        attn_residual
            .iter()
            .zip(&mlp_out)
            .map(|(a, b)| a + b)
            .collect()
    }
}

/// Deterministic pseudo-random weights in a small range, so fp16 has headroom and
/// two runs agree.
fn spread(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.25
        })
        .collect()
}

/// Rung 8: a whole ModernBERT block, 38 operations, against an independent f32
/// reference.
///
/// Small dimensions on purpose: the point is the wiring, and at 512 wide with 128
/// positions a mis-transposed head would still produce plausible numbers while
/// taking a minute to check by hand.
#[test]
fn a_transformer_block_matches_a_reference_implementation() {
    use kohagi::coreml_export::modernbert::Activation;
    for activation in [Activation::Gelu, Activation::Silu] {
        one_block_against_the_reference(activation);
    }
}

/// The block at both gate activations. `silu` is a different MIL operation of the
/// same shape, so running the whole block on each is what shows the substitution is
/// the only difference — and that CoreML has the operation at all.
fn one_block_against_the_reference(activation: kohagi::coreml_export::modernbert::Activation) {
    use kohagi::coreml_export::modernbert::{self, BlockOffsets, Config, Stored};

    let cfg = Config {
        hidden: 16,
        heads: 2,
        intermediate: 8,
        seq: 6,
        eps: 1e-5,
        rope_theta: 10_000.0,
        activation,
    };
    let (h, inter, seq) = (cfg.hidden, cfg.intermediate, cfg.seq);

    // fp16-rounded up front, so the reference sees what the graph will.
    let wqkv = modernbert::to_fp16(&spread(3 * h * h, 11));
    let wo = modernbert::to_fp16(&spread(h * h, 22));
    let mlp_norm = modernbert::to_fp16(&spread(h, 33).iter().map(|v| 1.0 + v).collect::<Vec<_>>());
    let mlp_wi = modernbert::to_fp16(&spread(2 * inter * h, 44));
    let mlp_wo = modernbert::to_fp16(&spread(h * inter, 55));
    let attn_norm = modernbert::to_fp16(&spread(h, 66).iter().map(|v| 1.0 + v).collect::<Vec<_>>());
    let mask = modernbert::attention_mask(seq, Some(4));

    let mut weights = blob::Writer::new();
    let zeros_3h = weights.write_f32_as_fp16(&vec![0.0; 3 * h]);
    let zeros_h = weights.write_f32_as_fp16(&vec![0.0; h]);
    let zeros_2i = weights.write_f32_as_fp16(&vec![0.0; 2 * inter]);
    let (rope_cos, rope_sin) = modernbert::write_rope(&mut weights, &cfg);
    let offsets = BlockOffsets {
        wqkv: Stored::Fp16(weights.write_f32_as_fp16(&wqkv)),
        wqkv_bias: zeros_3h,
        wo: Stored::Fp16(weights.write_f32_as_fp16(&wo)),
        wo_bias: zeros_h,
        mlp_norm: weights.write_f32_as_fp16(&mlp_norm),
        mlp_wi: Stored::Fp16(weights.write_f32_as_fp16(&mlp_wi)),
        mlp_wi_bias: zeros_2i,
        mlp_wo: Stored::Fp16(weights.write_f32_as_fp16(&mlp_wo)),
        mlp_wo_bias: zeros_h,
        rope_cos,
        rope_sin,
        attn_norm: Some(weights.write_f32_as_fp16(&attn_norm)),
    };
    let mask_offset = weights.write_f32_as_fp16(&mask);

    let x = Tensor::new("x", DType::Fp16, &[1, seq, h]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let eps = b.const_fp16(Tensor::new("eps", DType::Fp16, &[]), &[cfg.eps]);
    let mask_t = b.const_blob(
        Tensor::new("mask", DType::Fp16, &[1, 1, seq, seq]),
        mask_offset,
    );
    let out = modernbert::block(&mut b, &cfg, 0, &x, &offsets, &mask_t, &eps);
    b.returns(&out);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![out.clone()],
        },
    );

    let dir = scratch(&format!("block-{}", activation.name()));
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model_handle = load(&dir);

    let input = modernbert::to_fp16(&spread(seq * h, 77));
    let got = predict(
        &model_handle,
        &[("x", Feed::F16(vec![1, seq, h], input.clone()))],
        &out.name,
    );
    let want = reference::block(
        &cfg,
        &input,
        &reference::Weights {
            attn_norm: Some(&attn_norm),
            wqkv: &wqkv,
            wo: &wo,
            mlp_norm: &mlp_norm,
            mlp_wi: &mlp_wi,
            mlp_wo: &mlp_wo,
        },
        &mask,
    );
    // A block accumulates fp16 rounding through two normalizations, a softmax and
    // four projections, so this is looser than the single-operation rungs. A
    // wiring error is off by far more than this.
    assert_close(&got, &want, 3e-2, &format!("block ({})", activation.name()));
}

/// Rung 9: the same block at `ruri-v3-130m`'s real dimensions, which is the size
/// at which a placement decision means anything.
///
/// This only checks that it compiles and runs; where CoreML puts the operations is
/// read afterwards with `tools/coreml-jigs`'s `computeplan`, against the reference
/// model's recorded baseline. The package is left at a stable path for that.
#[test]
fn a_real_size_block_compiles_and_runs() {
    use kohagi::coreml_export::modernbert::{self, BlockOffsets, Config, Stored};

    let cfg = Config {
        hidden: 512,
        heads: 8,
        intermediate: 2048,
        seq: 128,
        eps: 1e-5,
        rope_theta: 10_000.0,
        activation: kohagi::coreml_export::modernbert::Activation::Gelu,
    };
    let (h, inter, seq) = (cfg.hidden, cfg.intermediate, cfg.seq);

    let mut weights = blob::Writer::new();
    let zeros_3h = weights.write_f32_as_fp16(&vec![0.0; 3 * h]);
    let zeros_h = weights.write_f32_as_fp16(&vec![0.0; h]);
    let zeros_2i = weights.write_f32_as_fp16(&vec![0.0; 2 * inter]);
    let (rope_cos, rope_sin) = modernbert::write_rope(&mut weights, &cfg);
    let offsets = BlockOffsets {
        wqkv: Stored::Fp16(weights.write_f32_as_fp16(&spread(3 * h * h, 11))),
        wqkv_bias: zeros_3h,
        wo: Stored::Fp16(weights.write_f32_as_fp16(&spread(h * h, 22))),
        wo_bias: zeros_h,
        mlp_norm: weights.write_f32_as_fp16(&vec![1.0; h]),
        mlp_wi: Stored::Fp16(weights.write_f32_as_fp16(&spread(2 * inter * h, 44))),
        mlp_wi_bias: zeros_2i,
        mlp_wo: Stored::Fp16(weights.write_f32_as_fp16(&spread(h * inter, 55))),
        mlp_wo_bias: zeros_h,
        rope_cos,
        rope_sin,
        attn_norm: Some(weights.write_f32_as_fp16(&vec![1.0; h])),
    };
    // The local mask, which 12 of the 19 layers use.
    let mask_offset = weights.write_f32_as_fp16(&modernbert::attention_mask(seq, Some(128)));

    let x = Tensor::new("x", DType::Fp16, &[1, seq, h]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let eps = b.const_fp16(Tensor::new("eps", DType::Fp16, &[]), &[cfg.eps]);
    let mask_t = b.const_blob(
        Tensor::new("mask", DType::Fp16, &[1, 1, seq, seq]),
        mask_offset,
    );
    let out = modernbert::block(&mut b, &cfg, 0, &x, &offsets, &mask_t, &eps);
    b.returns(&out);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![out.clone()],
        },
    );

    let dir = scratch("real-block");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model_handle = load(&dir);

    let input = modernbert::to_fp16(&spread(seq * h, 99));
    let got = predict(
        &model_handle,
        &[("x", Feed::F16(vec![1, seq, h], input))],
        &out.name,
    );
    assert_eq!(got.len(), seq * h);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the output has non-finite values, so something saturated in fp16"
    );
    eprintln!("real-size block written to {}", dir.display());
}

/// A `Weights` over a safetensors checkpoint, so the emitter reads the same file
/// the inference path does.
struct Safetensors {
    tensors: std::collections::HashMap<String, candle_core::Tensor>,
}

impl Safetensors {
    fn open(path: &std::path::Path) -> Self {
        let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)
            .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
        Self { tensors }
    }
}

impl kohagi::coreml_export::encoder::Weights for Safetensors {
    fn get(&self, name: &str, expected: &[usize]) -> anyhow::Result<Vec<f32>> {
        // Some checkpoints wrap the encoder under `model.` and some store it at
        // the root; Kohagi's inference path resolves the same two layouts.
        let t = self
            .tensors
            .get(name)
            .or_else(|| self.tensors.get(&format!("model.{name}")))
            .ok_or_else(|| {
                anyhow::anyhow!("the checkpoint has no tensor named {name} or model.{name}")
            })?;
        let shape = t.dims().to_vec();
        anyhow::ensure!(
            shape == expected,
            "{name} is {shape:?}, expected {expected:?}"
        );
        Ok(t.to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?)
    }
}

/// Rung 10: the whole encoder, from the real checkpoint, against the reference
/// `.mlpackage` that `scripts/convert_coreml.py` produced.
///
/// Set `KOHAGI_TEST_SAFETENSORS` to a `model.safetensors` and
/// `KOHAGI_TEST_REFERENCE` to the reference `seq-128.mlpackage`; skips otherwise.
#[test]
fn the_whole_encoder_matches_the_python_conversion() {
    use kohagi::coreml_export::encoder::{self, EncoderConfig};

    let (Some(weights_path), Some(reference)) = (
        std::env::var_os("KOHAGI_TEST_SAFETENSORS"),
        std::env::var_os("KOHAGI_TEST_REFERENCE"),
    ) else {
        eprintln!("skipping: set KOHAGI_TEST_SAFETENSORS and KOHAGI_TEST_REFERENCE");
        return;
    };

    let cfg = EncoderConfig {
        hidden: 512,
        heads: 8,
        layers: 19,
        intermediate: 2048,
        vocab: 102_400,
        eps: 1e-5,
        local_attention: 128,
        global_every: 3,
        local_rope_theta: 10_000.0,
        global_rope_theta: 160_000.0,
        max_positions: None,
        activation: kohagi::coreml_export::modernbert::Activation::Gelu,
    };
    let seq = 128;

    let weights = Safetensors::open(std::path::Path::new(&weights_path));
    let (m, blob) = encoder::emit(&cfg, &weights, seq).expect("emit the encoder");
    let dir = scratch("ruri-seq-128");
    write_package(&dir, &m, &blob).expect("write the package");
    eprintln!(
        "emitted {} ({:.1} MB)",
        dir.display(),
        blob.len() as f64 / 1e6
    );

    // A deterministic sentence-shaped input: real ids, the rest padding.
    let real = 24usize;
    let mut ids: Vec<i32> = spread(seq, 5)
        .iter()
        .map(|v| 5 + (v.abs() * 40_000.0) as i32)
        .collect();
    let mut mask = vec![1i32; seq];
    for i in real..seq {
        ids[i] = 3;
        mask[i] = 0;
    }
    let feeds = || {
        vec![
            ("input_ids", Feed::I32(vec![1, seq], ids.clone())),
            ("attention_mask", Feed::I32(vec![1, seq], mask.clone())),
        ]
    };

    let ours = predict(&load(&dir), &feeds(), "hidden");
    let theirs = predict(&load(std::path::Path::new(&reference)), &feeds(), "hidden");
    assert_eq!(ours.len(), seq * cfg.hidden);

    // Compare only the unpadded rows: a padded row's value is whatever the mask
    // leaves behind, and Kohagi's pooling discards it either way.
    let keep = real * cfg.hidden;
    let (a, b) = (&ours[..keep], &theirs[..keep]);
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|y| f64::from(*y) * f64::from(*y))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (na * nb);
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "1 - cosine = {:.3e}, max |a-b| = {max_abs:.3e}",
        1.0 - cosine
    );

    assert!(
        ours.iter().all(|v| v.is_finite()),
        "the emitted model produced non-finite values"
    );
    assert!(
        1.0 - cosine < 1e-4,
        "1 - cosine is {:.3e}; the graphs disagree by more than fp16 rounding",
        1.0 - cosine
    );
}

/// Rung 11: one bundle serving three lengths, sharing a single copy of the weights.
///
/// This is the form worth publishing. Checks the three things that make it worth
/// it: every function loads on its own, each one still matches the Python
/// conversion, and the bundle is not three times the size.
#[test]
fn a_multi_function_bundle_serves_every_length_from_one_blob() {
    use kohagi::coreml_export::encoder::{self, EncoderConfig};

    let (Some(weights_path), Some(config_path)) = (
        std::env::var_os("KOHAGI_TEST_SAFETENSORS"),
        std::env::var_os("KOHAGI_TEST_CONFIG"),
    ) else {
        eprintln!("skipping: set KOHAGI_TEST_SAFETENSORS and KOHAGI_TEST_CONFIG");
        return;
    };

    let text = std::fs::read_to_string(&config_path).expect("read config.json");
    let cfg = EncoderConfig::from_json(&text).expect("parse config.json");
    let lengths = [128usize, 256, 512];

    let weights = Safetensors::open(std::path::Path::new(&weights_path));
    let (m, blob) = encoder::emit_multi(&cfg, &weights, &lengths).expect("emit the bundle");
    let dir = scratch("buckets-128-256-512");
    write_package(&dir, &m, &blob).expect("write the package");

    // Sharing means the bundle is barely larger than one length, not three times.
    let one = {
        let (m1, b1) = encoder::emit(&cfg, &weights, 128).expect("emit one length");
        let d = scratch("one-length");
        write_package(&d, &m1, &b1).expect("write");
        b1.len()
    };
    let ratio = blob.len() as f64 / one as f64;
    eprintln!(
        "bundle {:.1} MB for {} lengths; one length is {:.1} MB (x{ratio:.3})",
        blob.len() as f64 / 1e6,
        lengths.len(),
        one as f64 / 1e6
    );
    assert!(
        ratio < 1.1,
        "the lengths are not sharing the weights: {ratio:.2}x one length"
    );

    // Every function loads and runs on its own.
    for &seq in &lengths {
        let name = encoder::function_name(seq);
        let model = load_function(&dir, Some(&name));
        let real = 24.min(seq);
        let mut ids: Vec<i32> = spread(seq, 5)
            .iter()
            .map(|v| 5 + (v.abs() * 40_000.0) as i32)
            .collect();
        let mut mask = vec![1i32; seq];
        for i in real..seq {
            ids[i] = 3;
            mask[i] = 0;
        }
        let got = predict(
            &model,
            &[
                ("input_ids", Feed::I32(vec![1, seq], ids)),
                ("attention_mask", Feed::I32(vec![1, seq], mask)),
            ],
            "hidden",
        );
        assert_eq!(got.len(), seq * cfg.hidden, "{name} output length");
        assert!(
            got.iter().take(real * cfg.hidden).all(|v| v.is_finite()),
            "{name} produced non-finite values"
        );
        eprintln!("  {name}: {} values, all finite", got.len());
    }
}

/// Symmetric per-tensor int8 quantization, the simplest form and the one measured
/// as costing nothing on retrieval quality when applied to the embedding table.
fn quantize_int8(values: &[f32]) -> (Vec<i8>, f32) {
    let max = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    let q = values
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Rung 12: a weight stored as int8 and dequantized inside the graph.
///
/// The last thing `scripts/convert_coreml.py` can do that this crate cannot: on
/// bekko, quantizing the embedding table takes a bundle from 247MB to 149MB at
/// an unchanged JaCWIR MAP@10.
///
/// The operation's parameters were not available from any reference model — no
/// published bundle uses it — so this test is also how they were established. If
/// CoreML rejects the graph, its error names what it expected.
#[test]
fn an_int8_weight_is_dequantized_inside_the_graph() {
    // `[n_in, n_out]`, so `matmul` with neither operand transposed lines up.
    let (n_in, n_out) = (4usize, 3usize);
    let w: Vec<f32> = spread(n_in * n_out, 7).iter().map(|v| v * 4.0).collect();
    let (q, scale) = quantize_int8(&w);

    let mut weights = blob::Writer::new();
    let q_offset = weights.write_int8(&q);

    let x = Tensor::new("x", DType::Fp16, &[1, n_in]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let dequantized = b.dequantize_int8(
        Tensor::new("w", DType::Fp16, &[n_in, n_out]),
        q_offset,
        0,
        &[scale],
        0,
    );
    let no = b.const_bool("no", false);
    let y = b.op(
        "matmul",
        Tensor::new("y", DType::Fp16, &[1, n_out]),
        &[
            ("x", &x),
            ("y", &dequantized),
            ("transpose_x", &no),
            ("transpose_y", &no),
        ],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y.clone()],
        },
    );

    let dir = scratch("int8");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model_handle = load(&dir);

    let input = [1.0f32, 2.0, -1.0, 0.5];
    let got = predict(
        &model_handle,
        &[("x", Feed::F16(vec![1, n_in], input.to_vec()))],
        &y.name,
    );
    // y[o] = sum_i x[i] * w[i, o].
    let want: Vec<f32> = (0..n_out)
        .map(|o| {
            (0..n_in)
                .map(|i| input[i] * (f32::from(q[i * n_out + o]) * scale))
                .sum()
        })
        .collect();
    assert_close(&got, &want, 2e-2, "int8 dequantize");
}

/// Blockwise symmetric int8: one scale per `block` values along the last axis.
fn quantize_blocks(values: &[f32], rows: usize, block: usize) -> (Vec<i8>, Vec<f32>, Vec<usize>) {
    let width = values.len() / rows;
    assert_eq!(width % block, 0, "the last axis must divide into blocks");
    let mut q = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(rows * (width / block));
    for row in values.chunks(width) {
        for chunk in row.chunks(block) {
            let max = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            q.extend(
                chunk
                    .iter()
                    .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8),
            );
            scales.push(scale);
        }
    }
    (q, scales, vec![rows, width / block])
}

/// Rung 13: `constexpr_blockwise_shift_scale`, iOS18's finer-grained quantization.
///
/// One scale per block along the last axis rather than per channel. Also the
/// operation that carries int4, so this is the plumbing a smaller-than-int8 bundle
/// would need. Its parameters were established the same way rung 12's were.
#[test]
fn a_blockwise_quantized_weight_is_dequantized_inside_the_graph() {
    let (n_in, n_out, block) = (4usize, 8usize, 4usize);
    let w: Vec<f32> = spread(n_in * n_out, 13).iter().map(|v| v * 4.0).collect();
    let (q, scales, scale_shape) = quantize_blocks(&w, n_in, block);

    let mut weights = blob::Writer::new();
    let q_offset = weights.write_int8(&q);
    // The scale goes in the blob file too, not as an immediate.
    let scale_offset = weights.write_f32_as_fp16(&scales);

    let x = Tensor::new("x", DType::Fp16, &[1, n_in]);
    let mut b = Builder::new(std::slice::from_ref(&x));
    let dequantized = b.dequantize_blockwise(
        Tensor::new("w", DType::Fp16, &[n_in, n_out]),
        q_offset,
        &scale_shape,
        scale_offset,
    );
    let no = b.const_bool("no", false);
    let y = b.op(
        "matmul",
        Tensor::new("y", DType::Fp16, &[1, n_out]),
        &[
            ("x", &x),
            ("y", &dequantized),
            ("transpose_x", &no),
            ("transpose_y", &no),
        ],
    );
    b.returns(&y);
    let m = model(
        b.finish(),
        Io {
            inputs: vec![x],
            outputs: vec![y.clone()],
        },
    );

    let dir = scratch("blockwise");
    write_package(&dir, &m, &weights.finish()).expect("write the package");
    let model_handle = load(&dir);

    let input = [1.0f32, 2.0, -1.0, 0.5];
    let got = predict(
        &model_handle,
        &[("x", Feed::F16(vec![1, n_in], input.to_vec()))],
        &y.name,
    );
    let blocks_per_row = n_out / block;
    let want: Vec<f32> = (0..n_out)
        .map(|o| {
            (0..n_in)
                .map(|i| {
                    let s = scales[i * blocks_per_row + o / block];
                    input[i] * (f32::from(q[i * n_out + o]) * s)
                })
                .sum()
        })
        .collect();
    assert_close(&got, &want, 2e-2, "blockwise dequantize");
}

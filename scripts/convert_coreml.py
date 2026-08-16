#!/usr/bin/env python3
"""Convert a ModernBERT sentence encoder to the CoreML layout Kohagi expects.

NOT the converter to use. `coreml-convert` (`--features coreml-export`) is, and it
is what the published models are made with:

    cargo run --release --bin coreml-convert --features coreml,coreml-export -- \\
        --model-id cl-nagoya/ruri-v3-130m --out-dir <out-dir> \\
        --sequence-lengths 64,128,256,512 --compiled

This script is kept as the independent second opinion that checked that one — a
conversion through PyTorch and coremltools, arriving at the same vectors by another
route. Do not publish what it writes without looking at the mask first. It traces
the model in fp32, where `transformers` blocks attention with
`torch.finfo(torch.float32).min`, and coremltools casts that constant to fp16 on the
way out; -3.4e38 has no fp16 spelling but `-inf` does. A bundle that blocks with
`-inf` returns NaN for the whole embedding of a padded input on the CPU compute unit
— see `BLOCKED` in `src/coreml_export/modernbert.rs`. Whether this path lands there
has not been measured; `grep -c -- -inf <compiled>/model.mil` after a conversion
answers it.

Kohagi's `--device coreml` backend runs the encoder on the Apple Neural Engine.
The ANE needs a *fixed-shape, batch=1* model, and one model per sequence length,
so this script emits one `seq-<N>.mlpackage` per bucket length plus the
tokenizer and config:

    <out-dir>/
      seq-128.mlpackage
      seq-256.mlpackage
      seq-512.mlpackage
      tokenizer.json
      config.json
      1_Pooling/config.json     # if the base model ships one
      compiled/                 # only with --compiled
        seq-128.mlmodelc
        seq-256.mlmodelc
        seq-512.mlmodelc

With --multi-function the lengths land in one bundle instead, as one CoreML
function each, sharing a single copy of the weights:

    <out-dir>/
      buckets-128-256-512.mlpackage    # functions seq_128 / seq_256 / seq_512
      tokenizer.json
      config.json
      1_Pooling/config.json
      compiled/
        buckets-128-256-512.mlmodelc

That is the form to upload: on bekko-embedding-v1-a25m it is 248MB against
740MB for the three separate packages, with bit-identical output and latency
within a percent. Kohagi reads either layout.

Point Kohagi at it locally:

    kohagi --device coreml --coreml-dir <out-dir>

or upload <out-dir> to a Hugging Face repo and use --coreml-model-id (the
converted model is a derivative; keep the base model's license — ruri-v3 is
Apache-2.0 — and set `base_model:` in the model card).

Requirements (a throwaway venv is fine):

    uv venv --python 3.12 .venv && . .venv/bin/activate
    uv pip install torch "transformers==4.48.3" coremltools numpy

transformers must be 4.48.x: 5.x's masking_utils does not trace.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel


def patch_int_op():
    """coremltools' `int` op does `int(x.val)`, which raises under numpy>=2 on a
    1-element 1-D array (`int(np.array([5]))`). ModernBERT's traced graph hits
    it; reimplement the op to flatten first."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as tops

    def patched_int(context, node):
        x = context[node.inputs[0]]
        if x.val is not None:
            res = mb.const(val=int(np.asarray(x.val).reshape(-1)[0]), name=node.name)
        else:
            res = mb.cast(x=x, dtype="int32", name=node.name)
        context.add(res)

    tops._TORCH_OPS_REGISTRY.set_func_by_name(patched_int, "int")


class Encoder(torch.nn.Module):
    """Expose a clean (input_ids, attention_mask) -> last_hidden_state forward.
    Pooling and L2 normalization stay in Kohagi (Rust)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def source_fingerprint(model_id):
    """sha256 of the checkpoint's weights, for the bundle's metadata.

    A converted bundle holds fp16 copies of the weights, so it cannot be hashed
    into anything comparable with the checkpoint it came from; the checkpoint's
    own digest is the only fingerprint it can carry. Kohagi's Rust converter
    records the same key, and reports it back under `--print-model-info`.

    `None` when the weights are not one cached file — sharded checkpoints, or a
    local directory this cannot resolve. An absent key reads as "unknown",
    which is true; a guessed one would not be.
    """
    import hashlib

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "model.safetensors")
    except Exception as e:
        print(f"  no single model.safetensors ({type(e).__name__}); no source fingerprint recorded")
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(model_id):
    """The metadata every bucket carries: what it was converted from, by what.

    Deliberately not `graph_version`: that names the graph Kohagi's own emitter
    builds, and this script converts through coremltools instead. Claiming it
    would make two different graphs answer to one version.
    """
    entries = {
        "com.github.takahashim.kohagi.emitter": "convert_coreml.py",
        "com.github.takahashim.kohagi.source": model_id,
    }
    sha = source_fingerprint(model_id)
    if sha:
        entries["com.github.takahashim.kohagi.source_sha256"] = sha
    return entries


def convert_bucket(enc, seq, out_path, metadata=None):
    import coremltools as ct

    ids = torch.randint(5, 1000, (1, seq), dtype=torch.long)
    mask = torch.ones((1, seq), dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(enc, (ids, mask), strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="hidden")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    for key, value in (metadata or {}).items():
        mlmodel.user_defined_metadata[key] = value
    mlmodel.save(str(out_path))
    print(f"  saved {out_path.name}")


def merge_functions(out_dir, buckets):
    """Merge the per-length packages into one multi-function bundle and return it.

    Each length becomes a `seq_<N>` function, and `save_multifunction`
    deduplicates the weights they share — which for a large-vocabulary encoder is
    nearly everything, since only the fixed input shape differs. The per-length
    packages are removed afterwards so the directory holds one form of each
    bucket; Kohagi reads the lengths back out of the bundle's name.
    """
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    desc = MultiFunctionDescriptor()
    for seq in buckets:
        desc.add_function(
            str(out_dir / f"seq-{seq}.mlpackage"),
            src_function_name="main",
            target_function_name=f"seq_{seq}",
        )
    desc.default_function_name = f"seq_{buckets[0]}"

    merged = out_dir / ("buckets-" + "-".join(str(s) for s in buckets) + ".mlpackage")
    if merged.exists():
        shutil.rmtree(merged)
    print(f"merging {len(buckets)} lengths into {merged.name} ...")
    save_multifunction(desc, str(merged))
    for seq in buckets:
        shutil.rmtree(out_dir / f"seq-{seq}.mlpackage")
    return merged


def compile_beside(out_dir, bundle):
    """Compile `bundle` into `compiled/<name>.mlmodelc` next to it."""
    from coremltools.models.utils import compile_model

    compiled_dir = out_dir / "compiled"
    compiled_dir.mkdir(exist_ok=True)
    mlmodelc = compiled_dir / (bundle.stem + ".mlmodelc")
    if mlmodelc.exists():
        shutil.rmtree(mlmodelc)
    shutil.copytree(compile_model(str(bundle)), mlmodelc)
    print(f"  compiled compiled/{mlmodelc.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="cl-nagoya/ruri-v3-130m", help="HF model to convert")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory for the CoreML layout")
    ap.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="fixed sequence lengths to emit (one model each)",
    )
    ap.add_argument(
        "--compiled",
        action="store_true",
        help="also emit a compiled .mlmodelc beside each .mlpackage. Kohagi loads "
        "the .mlmodelc directly and falls back to the .mlpackage if it can't. "
        "Doubles the output size; without it Kohagi compiles the .mlpackage on "
        "first use (~20s per bucket) and caches the result, so only the first run "
        "pays.",
    )
    ap.add_argument(
        "--multi-function",
        action="store_true",
        help="emit one buckets-<N>-<N>....mlpackage holding every length as a "
        "CoreML function instead of one .mlpackage per length. The lengths then "
        "share a single copy of the weights: on bekko-embedding-v1-a25m, 3 "
        "buckets go from 740MB to 248MB with bit-identical output and no "
        "measurable change in latency. Needs macOS 15 to load.",
    )
    args = ap.parse_args()

    patch_int_op()
    print(f"loading {args.model_id} ...")
    model = AutoModel.from_pretrained(args.model_id, attn_implementation="eager").eval()
    enc = Encoder(model)

    buckets = sorted(args.buckets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Stamped on each bucket as it is written. `--multi-function` merges the
    # buckets afterwards, and whether the merge carries the metadata across is
    # coremltools' business; Kohagi's own converter is the one that guarantees
    # it, and a bundle without the key is reported as unknown rather than
    # assumed.
    stamp = provenance(args.model_id)
    for seq in buckets:
        out = args.out_dir / f"seq-{seq}.mlpackage"
        if out.exists():
            shutil.rmtree(out)
        print(f"converting seq={seq} ...")
        convert_bucket(enc, seq, out, stamp)

    bundles = [args.out_dir / f"seq-{seq}.mlpackage" for seq in buckets]
    if args.multi_function:
        bundles = [merge_functions(args.out_dir, buckets)]
    if args.compiled:
        for bundle in bundles:
            compile_beside(args.out_dir, bundle)

    # Copy tokenizer.json and config.json next to the buckets, from the HF cache.
    from huggingface_hub import hf_hub_download

    for fname in ("tokenizer.json", "config.json"):
        src = hf_hub_download(args.model_id, fname)
        shutil.copy(src, args.out_dir / fname)
        print(f"  copied {fname}")

    # And the declared pooling, so Kohagi reads it from the converted directory
    # exactly as it would from the base checkpoint. Without it Kohagi falls back
    # to mean and warns that this may not be a sentence-embedding model at all —
    # true of a reranker, alarming for a converted encoder that is fine.
    # Copy rather than symlink: the HF cache stores a tree of symlinks, and this
    # directory is meant to be uploadable as-is.
    try:
        src = hf_hub_download(args.model_id, "1_Pooling/config.json")
    except Exception as e:  # a model may genuinely ship none
        print(f"  no 1_Pooling/config.json ({type(e).__name__}); Kohagi will warn and use mean")
    else:
        (args.out_dir / "1_Pooling").mkdir(exist_ok=True)
        shutil.copy(src, args.out_dir / "1_Pooling" / "config.json")
        print("  copied 1_Pooling/config.json")

    print(f"\ndone -> {args.out_dir}")
    print(f"try: kohagi --device coreml --coreml-dir {args.out_dir} --text '瑠璃も玻璃も照らせば光る'")


if __name__ == "__main__":
    main()

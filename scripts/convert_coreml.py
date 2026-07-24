#!/usr/bin/env python3
"""Convert a ModernBERT sentence encoder to the CoreML layout Kohagi expects.

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
      compiled/                 # only with --compiled
        seq-128.mlmodelc
        seq-256.mlmodelc
        seq-512.mlmodelc

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


def convert_bucket(enc, seq, out_path):
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
    mlmodel.save(str(out_path))
    print(f"  saved {out_path.name}")


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
        help="also emit a compiled seq-<N>.mlmodelc beside each .mlpackage. "
        "Kohagi then loads the .mlmodelc directly (no per-run compile) and "
        "falls back to the .mlpackage if it can't. Doubles the output size.",
    )
    args = ap.parse_args()

    patch_int_op()
    print(f"loading {args.model_id} ...")
    model = AutoModel.from_pretrained(args.model_id, attn_implementation="eager").eval()
    enc = Encoder(model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seq in sorted(args.buckets):
        out = args.out_dir / f"seq-{seq}.mlpackage"
        if out.exists():
            shutil.rmtree(out)
        print(f"converting seq={seq} ...")
        convert_bucket(enc, seq, out)
        if args.compiled:
            from coremltools.models.utils import compile_model

            compiled_dir = args.out_dir / "compiled"
            compiled_dir.mkdir(exist_ok=True)
            mlmodelc = compiled_dir / f"seq-{seq}.mlmodelc"
            if mlmodelc.exists():
                shutil.rmtree(mlmodelc)
            shutil.copytree(compile_model(str(out)), mlmodelc)
            print(f"  compiled compiled/{mlmodelc.name}")

    # Copy tokenizer.json and config.json next to the buckets, from the HF cache.
    from huggingface_hub import hf_hub_download

    for fname in ("tokenizer.json", "config.json"):
        src = hf_hub_download(args.model_id, fname)
        shutil.copy(src, args.out_dir / fname)
        print(f"  copied {fname}")

    print(f"\ndone -> {args.out_dir}")
    print(f"try: kohagi --device coreml --coreml-dir {args.out_dir} --text '瑠璃も玻璃も照らせば光る'")


if __name__ == "__main__":
    main()

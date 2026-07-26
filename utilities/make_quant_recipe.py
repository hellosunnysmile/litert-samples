#!/usr/bin/env python3
"""Generate a mixed-precision quantization recipe for a decoder-only LLM.

Uniform quantization forces one answer for a whole model, and on Qwen3-0.6B that answer
is bad in both directions: int4 everywhere loses 7 of 11 correct answers, int8
everywhere leaves 38% of the file size and 6% of the speed on the table. A recipe that
keeps 4-bit weights for the bulk and lifts specific layers to 8 bits beats both.

    python make_quant_recipe.py down8 > recipe.json
    python make_quant_recipe.py edges8 --layers 28 > recipe.json
    python make_quant_recipe.py down8 --bulk-bits 2      # experiment

Feed the result to either stage of the pipeline:

    etf produce <model> --quantize ./recipe.json
    etf optimize model.litertlm --recipe ./recipe.json
    # or upstream directly:
    python -m litert_torch.generative.export_hf --model=<hf-id> \\
        --output_dir=out --quantization_recipe=./recipe.json

This has no dependencies — it prints JSON in the schema `ai-edge-quantizer` loads.

**Verify the result**, always: `inspect_quantization.py <model>.tflite --match <regex>`
reports the precision each layer actually got. A regex that matches nothing produces a
recipe that silently does nothing.

Measured on Qwen3-0.6B (GPU, greedy decoding, 11-case suite):

    policy    size     decode        quality
    int8      585 MB    90.2 tok/s     9/11
    down8     363 MB    95.7 tok/s    10/11   <- dominates int8
    attn8     402 MB    92.0 tok/s     9/11
    edges8    353 MB    95.5 tok/s     6/11
    int4      325 MB   101.6 tok/s     5/11
"""

from __future__ import annotations

import argparse
import json
import sys

ALGORITHM = "min_max_uniform_quantize"

#: Scope regexes. A converted LLM keeps the PyTorch module path in its tensor names, and
#: these attribute names (`q_proj`, `down_proj`, …) are shared across HuggingFace
#: decoder LMs — Llama, Qwen, Mistral, Phi — so the same regex addresses all of them.
ATTENTION = r"Linear_(q|k|v|o)_proj"
DOWN = r"Linear_down_proj"
MLP = r"Linear_(gate|up|down)_proj"

POLICIES = {
    "down8": "int4 everywhere except the down-projections — the widest fan-in matmuls",
    "attn8": "int4 everywhere except the attention projections",
    "mlp8": "int4 everywhere except the MLP projections",
    "edges8": "int4 everywhere except the first and last blocks (needs --layers)",
    "embed8": "int4 everywhere except the embedding table (small vocabularies only)",
    "uniform": "no overrides — every quantizable op at --bulk-bits",
}


def rule(regex: str, operation: str, num_bits: int, granularity: str) -> dict:
    return {
        "regex": regex,
        "operation": operation,
        "algorithm_key": ALGORITHM,
        "op_config": {
            "weight_tensor_config": {
                "num_bits": num_bits,
                "symmetric": True,
                "granularity": granularity,
                "dtype": "INT",
            },
            "compute_precision": "INTEGER",
            "explicit_dequantize": False,
            "skip_checks": False,
            "min_weight_elements": 0,
        },
    }


def granularity_for(num_bits: int) -> str:
    """4-bit weights need a scale per 32 values to survive; 8-bit is fine per channel.

    This is not a preference. The quantizer *rejects* 8-bit blockwise, and one scale per
    channel at 4 bits took Qwen3-0.6B from 9/11 to 2/11.
    """
    return "BLOCKWISE_32" if num_bits <= 4 else "CHANNELWISE"


def edge_regex(layers: int, keep: int = 2) -> str:
    indices = sorted(set(list(range(keep)) + list(range(max(layers - keep, keep), layers))))
    return r"DecoderLayer_(" + "|".join(str(i) for i in indices) + r")/"


def build(policy: str, bulk_bits: int = 4, protect_bits: int = 8,
          layers: int | None = None) -> list[dict]:
    """Bulk rule first, overrides after: rules are applied in order and the last wins."""
    if policy not in POLICIES:
        raise SystemExit(f"unknown policy '{policy}' (have {', '.join(POLICIES)})")

    # The bulk rule must cover *every* quantizable op ("*"). Naming FULLY_CONNECTED
    # instead leaves the rest in float, and the artifact comes out larger than an int8
    # build — which is how this was discovered.
    recipe = [rule(".*", "*", bulk_bits, granularity_for(bulk_bits))]

    protect = {
        "down8": DOWN,
        "attn8": ATTENTION,
        "mlp8": MLP,
    }.get(policy)
    if protect:
        recipe.append(rule(protect, "FULLY_CONNECTED", protect_bits,
                           granularity_for(protect_bits)))
    elif policy == "edges8":
        if not layers:
            raise SystemExit("'edges8' needs --layers (the model's num_hidden_layers)")
        recipe.append(rule(edge_regex(layers), "FULLY_CONNECTED", protect_bits,
                           granularity_for(protect_bits)))
    elif policy == "embed8":
        # Costly: on a 151k vocabulary, lifting only the embedding to 8 bits made the
        # artifact *larger* than quantizing the whole model to 8 bits.
        recipe.append(rule(".*", "EMBEDDING_LOOKUP", protect_bits,
                           granularity_for(protect_bits)))
    return recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", nargs="?", choices=sorted(POLICIES),
                        help="Which layers to keep at 8 bits.")
    parser.add_argument("--bulk-bits", type=int, default=4, help="Precision for the bulk.")
    parser.add_argument("--protect-bits", type=int, default=8,
                        help="Precision for the protected layers.")
    parser.add_argument("--layers", type=int, default=None,
                        help="Model depth (num_hidden_layers), for 'edges8'.")
    parser.add_argument("--list", action="store_true", help="Describe the policies.")
    args = parser.parse_args(argv)

    if args.list or not args.policy:
        for name, summary in POLICIES.items():
            print(f"  {name:<9} {summary}")
        return 0 if args.list else 1

    json.dump(build(args.policy, args.bulk_bits, args.protect_bits, args.layers),
              sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

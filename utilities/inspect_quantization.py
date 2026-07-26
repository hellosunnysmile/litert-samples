#!/usr/bin/env python3
"""Report what precision a converted LiteRT model actually uses, per layer.

A quantization recipe says what *should* happen. This says what did. It exists because
a recipe that silently does nothing looks exactly like one that works — until an
artifact comes out bigger than the int8 build you were trying to beat.

    python inspect_quantization.py model.tflite
    python inspect_quantization.py model.litertlm --by-layer
    python inspect_quantization.py model.litertlm --match Linear_down_proj

Run it with the interpreter that has `ai-edge-quantizer` installed (the conversion
environment, not the serving one).

Scope names are the hook everything else hangs on: a converted LLM keeps the PyTorch
module path in its tensor names, e.g.

    …/LlamaDecoderLayer_23/LlamaAttention_self_attn/torch.nn.modules.linear.Linear_q_proj;

which is what quantization recipes match with regexes, and what `--by-layer` groups by.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

try:
    from ai_edge_quantizer.utils import litertlm_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fb
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit(
        "ai-edge-quantizer is required. Run this with the conversion environment's "
        "interpreter, e.g. ~/.etf/produce-venv/bin/python"
    )

#: Ops whose weights are worth reporting; everything else carries no meaningful bytes.
WEIGHT_OPS = ("FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D", "DEPTHWISE_CONV_2D",
              "BATCH_MATMUL", "TRANSPOSE_CONV")

#: Below this a buffer is a bias or a shape constant, not a weight matrix.
MIN_WEIGHT_BYTES = 100_000

#: The part of a scope name a human cares about: which block, which projection.
LAYER_PATTERN = re.compile(r"(DecoderLayer_\d+|Linear_[a-z_0-9]+|Embedding_[a-z_]+)")


def read(path: str):
    """Read a `.tflite`, or the model inside a `.litertlm` bundle.

    A served artifact is nearly always the bundle, so requiring the raw flatbuffer
    would mean this tool can't inspect the thing you actually shipped.
    """
    if not path.endswith(".litertlm"):
        return fb.read_model(path)
    bundle = litertlm_utils.LiteRTLMFile(path)
    for index in range(len(bundle.sections)):
        model = bundle.read_model(index)
        if model is not None:
            return model
    sys.exit(f"no TFLite model found inside '{path}'")


def op_name(model, operator) -> str:
    code = model.operatorCodes[operator.opcodeIndex]
    builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
    return str(fb.TFL_OP_CODE_TO_NAME.get(builtin, "UNKNOWN")).split(".")[-1]


def weights(model, subgraph):
    """Yield (op_name, scope, dtype, bytes) for every weight tensor in the subgraph."""
    for operator in subgraph.operators:
        name = op_name(model, operator)
        if name not in WEIGHT_OPS:
            continue
        scope = fb.get_op_scope(operator, subgraph.tensors)
        for index in operator.inputs:
            if index == -1:
                continue
            tensor = subgraph.tensors[index]
            data = model.buffers[tensor.buffer].data if tensor.buffer else None
            if data is None or len(data) < MIN_WEIGHT_BYTES:
                continue
            dtype = fb.TENSOR_CODE_TO_TYPE.get(tensor.type, str(tensor.type))
            yield name, scope, dtype, len(data)


def label(scope: str) -> str:
    """Shorten a scope to the parts that identify the layer."""
    found = LAYER_PATTERN.findall(scope)
    return "/".join(dict.fromkeys(found)) or scope[-60:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Path to a .tflite or .litertlm file.")
    parser.add_argument("--subgraph", type=int, default=0,
                        help="Which subgraph to read (0 = the prefill signature).")
    parser.add_argument("--by-layer", action="store_true",
                        help="List every weight tensor instead of a summary.")
    parser.add_argument("--match", default=None,
                        help="Only report layers whose scope matches this regex — the "
                             "same expression a quantization recipe would use.")
    args = parser.parse_args(argv)

    model = read(args.model)
    if args.subgraph >= len(model.subgraphs):
        print(f"model has {len(model.subgraphs)} subgraph(s)", file=sys.stderr)
        return 1
    subgraph = model.subgraphs[args.subgraph]
    name = subgraph.name.decode() if subgraph.name else f"#{args.subgraph}"

    pattern = re.compile(args.match) if args.match else None
    rows = [row for row in weights(model, subgraph)
            if pattern is None or pattern.search(row[1])]
    if not rows:
        print(f"No weight tensors matched in subgraph '{name}'."
              + (" The regex matched nothing — a recipe using it would do nothing."
                 if pattern else ""))
        return 1

    if args.by_layer:
        for op, scope, dtype, size in rows:
            print(f"  {op:<18} {dtype:<7} {size/1048576:7.1f} MB  {label(scope)}")
        print()

    tally: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for op, _scope, dtype, size in rows:
        tally[(op, dtype)].append(size)
    total = sum(size for _, _, _, size in rows)

    print(f"subgraph '{name}' — {len(rows)} weight tensors, {total/1048576:.0f} MB")
    for (op, dtype), sizes in sorted(tally.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {op:<18} {dtype:<7} {len(sizes):>4} tensors  "
              f"{sum(sizes)/1048576:7.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

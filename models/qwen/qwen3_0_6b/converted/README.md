# Qwen3-0.6B → LiteRT-LM

Recipes and measurements for converting `Qwen/Qwen3-0.6B` to a `.litertlm` that
LiteRT-LM serves, including the quantization that came out ahead.



## The recommended build

`qwen3_0_6b_int4_down8` — 4-bit blockwise weights everywhere except the 27
down-projections, which stay at 8 bits.

| | Size | Decode | TTFT | Quality |
| :-- | --: | --: | --: | --: |
| **int4 + int8 down-projections** | **363 MB** | **95.7 tok/s** | 514 ms | **10/11** |
| all int8 | 585 MB | 90.2 tok/s | 500 ms | 9/11 |
| all int4, blockwise-32 | 325 MB | 101.6 tok/s | 502 ms | 5/11 |
| all int4, channelwise | 301 MB | 113.3 tok/s | 521 ms | 2/11 |

M-series laptop GPU (LiteRT-LM resident engine), greedy decoding, 11-case capability
suite. It is smaller, faster **and** no less accurate than the uniform int8 build — the
only build measured here that dominates another on every axis.

For reference, `litert-community/Qwen3-0.6B`'s `qwen3_0_6b_mixed_int4` runs 138.7 tok/s
at 9/11 on the same machine, so there is still headroom this recipe doesn't reach.

## Reproduce it

```bash
# 1. Conversion environment (once). Nightly is required; do not install TensorFlow.
python3.11 -m venv ~/.etf/produce-venv
~/.etf/produce-venv/bin/pip install --pre litert-torch-nightly

# 2. Convert + quantize
etf produce Qwen/Qwen3-0.6B --cache-length 1024 --quantize mixed_int4_down8
#   equivalently, without ETF:
#   ~/.etf/produce-venv/bin/python -m litert_torch.generative.export_hf \
#       --model=Qwen/Qwen3-0.6B --output_dir=./out --cache_length=1024 \
#       --quantization_recipe=./qwen3_0_6b_int4_down8.quant.json

# 3. Check the recipe did what it claims
~/.etf/produce-venv/bin/python utilities/inspect_quantization.py out/model.tflite
#   FULLY_CONNECTED   INT4   164 tensors  163.0 MB
#   FULLY_CONNECTED   INT8    27 tensors   81.0 MB
#   EMBEDDING_LOOKUP  INT4     1 tensors   74.2 MB

# 4. Measure both axes before trusting it
etf bench Qwen3-0.6B-produced --accelerator gpu
etf eval  Qwen3-0.6B-produced
```

Conversion takes ~1.3 minutes and needs several GB of RAM.

## Files

| File | What it is |
| :-- | :-- |
| `qwen3_0_6b_int4_down8.quant.json` | The quantization recipe: 4-bit bulk, 8-bit `Linear_down_proj`. Feed to `--quantization_recipe` or `etf optimize --recipe`. |
| `qwen3_0_6b_int4_down8.recipe.json` | The full production receipt — source, quantization, cache length, prefill lengths. `etf produce <this file>` replays it. |

## Why the down-projections

Protecting them beat protecting the 110 attention projections (402 MB, 92.0 tok/s, 9/11)
and beat protecting whole blocks at the edges of the stack (353 MB, 95.5 tok/s, 6/11).
The layers that can't take 4-bit weights here are the widest fan-in matmuls — not
attention, and not position in the stack.

Don't assume that transfers. Re-run the comparison for a different model:
`etf produce <model> --sweep mixed`.

## Known limits

- The recipe is regex-based over scope names (`Linear_down_proj`), which works because
  the conversion keeps PyTorch module paths in tensor names. A model whose attribute
  names differ needs a different regex — check with
  `inspect_quantization.py --match <regex>` before trusting it.
- Numbers are for `cache_length=1024` and one prefill signature (128). Larger context or
  more signatures changes size and TTFT.

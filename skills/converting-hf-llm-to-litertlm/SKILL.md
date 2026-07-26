---
name: converting-hf-llm-to-litertlm
description: >-
  Converts a HuggingFace decoder-only LLM (Qwen, Llama, SmolLM, Gemma, Phi) into a
  servable .litertlm for LiteRT-LM, including the conversion environment. Use when
  producing an on-device LLM artifact from a PyTorch checkpoint, or when a
  litert-torch export fails to start. Don't use for vision-only models or for
  quantization choices — see the choosing-quantization skill for those.
---

# Converting a HuggingFace LLM to `.litertlm`

The conversion itself is one call. Everything that goes wrong goes wrong in the
environment, so build that first and exactly this way.

## 1. Build a separate conversion environment — never install into your serving env

`litert-torch` pins **older** `ai-edge-litert` / `ai-edge-quantizer` builds than the
`litert-lm` serving stack. Installing it beside a runtime downgrades the runtime. It
also drags in ~3 GB of PyTorch that a serving process never needs.

```bash
python3.11 -m venv ~/.etf/produce-venv          # 3.11: torchao's typing breaks on 3.14
~/.etf/produce-venv/bin/pip install --pre litert-torch-nightly
```

Three rules that are not negotiable:

- **Nightly, not stable.** `litert-torch` 0.9.1's KV-cache layer doesn't implement
  `get_max_length`, which `transformers` 5.x made abstract, so *every* export dies before
  tracing with `TypeError: Can't instantiate abstract class LiteRTLMCacheLayer`.
- **`--pre` is required** — `litert-converter` only ships pre-releases.
- **Do NOT install TensorFlow.** It and `litert-converter` each link their own LLVM;
  loading both aborts the process with
  `CommandLine Error: Option 'info-output-file' registered more than once`. If something
  demands `tensorflow.lite.python.schema_py_generated`, alias `ai-edge-litert`'s copy
  instead (see the reauthoring-a-model skill).

macOS note: upstream's README says Linux only. **It works on macOS arm64** —
`litert-converter` ships `macosx_12_0_arm64` wheels, and a 0.6B model converts in ~1.5
minutes on an M-series laptop.

## 2. Convert

```bash
~/.etf/produce-venv/bin/python -m litert_torch.generative.export_hf \
    --model=Qwen/Qwen3-0.6B \
    --output_dir=./out \
    --quantization_recipe=dynamic_wi8_afp32 \
    --cache_length=1024
```

Or, with ETF driving it (adds a receipt and registers the result):

```bash
etf produce Qwen/Qwen3-0.6B --cache-length 1024 --quantize dynamic_int8
```

The output is `out/model.litertlm` — weights, tokenizer, and chat template in one
container. **Always pass `--cache_length`**: it is the served context window, it costs
memory, and the default (4096) is usually more than a small model needs.

## 3. Sanity-check before you trust it

An artifact that loads is not an artifact that works. In order:

```bash
etf bench <model-id>     # does it load on the accelerator you want, and how fast
etf eval  <model-id>     # does it still answer correctly
```

A produced model should stay **staged** until both agree. Producing a model earns it no
trust — see the measuring-model-changes skill for why the eval must be greedy.

## 4. Failure signatures worth recognizing

| What you see | What it means |
| :-- | :-- |
| `Can't instantiate abstract class LiteRTLMCacheLayer` | stable litert-torch against transformers 5.x — use the nightly |
| `Option 'info-output-file' registered more than once` | TensorFlow and litert-converter in one process — uninstall TF |
| `expected scalar type torch.float32 but found torch.float16` | `experimental_use_fp16` doesn't trace on this model; drop it |
| An untraceable op deep in the model's own Python | the tracer can't follow this architecture — re-author it instead |

## 5. What to keep

Conversion is minutes and several GB; treat each artifact as a build product with a
receipt. Record, next to the artifact: the source model id, the exact quantization
recipe (or the generated recipe JSON), `cache_length`, prefill lengths, and the
converter version. `etf produce` writes `<artifact>.recipe.json` for this and can replay
it (`etf produce <recipe>.json`). Without it you cannot tell two 600 MB files apart six
weeks later.

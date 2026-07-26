---
name: reauthoring-a-model
description: >-
  Rebuilds a model from litert-torch's Generative API layers instead of tracing the
  HuggingFace module, and verifies it against the source checkpoint. Use when the tracer
  can't follow a model, or when you need graph-shape control (KV cache layout, mask
  handling) that tracing doesn't expose. Don't use when a plain export works and you
  only want speed — tracing won the GPU on every model measured here.
---

# Re-authoring a model

Two different things get called re-authoring. Keep them apart:

- **litert-torch Generative API** (PyTorch): rebuild the architecture from mobile-optimized
  layers, load the checkpoint's weights into it, convert. This skill.
- **LiteRT Tensor API** (`LiteRT/tensor`, C++): build the graph directly with
  `FullyConnected` / `Softmax` / `Transpose`, `ModelFactory::AddSignature`, `Save` — no
  PyTorch, no conversion. The only path to an architecture nobody has written down yet.

## When it's worth it

- **Coverage**: the tracer can't follow the model's Python. This is the escape hatch.
- **Control**: it exposes graph-shape options tracing doesn't — KV cache layout, mask as
  input, decode batch size. Those decide which GPU kernels are reachable (see
  making-a-model-run-on-gpu).

It is **not** a speedup for models that trace cleanly: measured on SmolLM2-135M and
Qwen3-0.6B, the traced build won the GPU by 32–42%.

```bash
etf produce --list-architectures                       # what upstream ships builders for
etf produce HuggingFaceTB/SmolLM2-135M-Instruct --author smollm2-135m --verify
etf produce Qwen/Qwen3-0.6B --author qwen3-0.6b --sweep kvcache
```

Upstream's catalogue lives in `litert_torch.generative.examples.<module>:<builder>`;
anything not in it is per-model work you write yourself.

## Verify against the checkpoint — always, and before converting

Re-authoring is hand-written code asserting "this is that model". An eval suite cannot
tell a faithful port from one that merely answers plausibly. Compare logits in PyTorch
first, and **report the deviation, not a verdict**:

```
[verify] max |logit difference| = 0.000415 (tolerance 0.001)
[verify] top-1 token agrees: True
```

SmolLM2-135M re-authors to within 4.15e-4 with the same top-1 token — faithful, but
outside upstream's 1e-4 default because the checkpoint is bfloat16 while the re-authored
model builds in float32. A bare "FAILED" would have read as a broken architecture; the
number says which knob to turn (`--set atol=1e-3`).

Don't gate on generated text: HuggingFace's generation loop honours EOS and the
re-authored wrapper runs to the token budget, so the two disagree for reasons that have
nothing to do with the weights.

## The four things that break, in the order you'll hit them

1. **`ModuleNotFoundError: tensorflow`** — the layers import
   `tensorflow.lite.python.schema_py_generated` for the TFLite flatbuffer schema, and
   nothing else from TensorFlow. **Do not install TensorFlow** (it and litert-converter
   each link LLVM and abort the process). Alias `ai-edge-litert`'s identical module:

   ```python
   from ai_edge_litert import schema_py_generated as schema
   tf = types.ModuleType("tensorflow"); lite = types.ModuleType("tensorflow.lite")
   python = types.ModuleType("tensorflow.lite.python")
   python.schema_py_generated = schema; lite.python = python; tf.lite = lite
   for name, module in (("tensorflow", tf), ("tensorflow.lite", lite),
                        ("tensorflow.lite.python", python)):
       # torch.dynamo walks sys.modules and calls find_spec on every entry, which
       # raises on a module whose __spec__ is None.
       module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
       module.__path__ = []
       sys.modules[name] = module
   sys.modules["tensorflow.lite.python.schema_py_generated"] = schema
   ```

2. **`AssertionError: Mask cache must be built`** — the builder defaults
   `mask_cache_size=0`. Pass the context length, or set `mask_as_input=True` and keep 0.
   One of the two must be true.

3. **`BFloat16 did not match Float`** during verification — load the HuggingFace model
   with `.float()` before comparing.

4. **`got multiple values for keyword argument 'context_length'`** — `convert_to_litert`
   already passes it to the bundler. This one throws away a finished conversion at the
   last step, so check it before starting a long run.

## Feed the checkpoint's prompting contract into the bundle

A re-authored artifact with the wrong chat template or stop tokens scores badly for
reasons that have nothing to do with the graph, and the comparison against a traced
build becomes meaningless. Pass them explicitly:

```python
converter.convert_to_litert(
    model, output_path=out, output_name_prefix="model",
    prefill_seq_len=[128], kv_cache_max_len=1024,
    quantize="dynamic_int8", output_format="litertlm",
    hf_tokenizer_model_path=f"{checkpoint}/tokenizer.json",
    jinja_prompt_template=tokenizer.chat_template,
    stop_token_ids=sorted(set(stop_ids)),
    export_config=export_config,
)
```

Note the quantization vocabulary differs from the tracing path: this converter takes
`none`, `dynamic_int8`, `weight_only_int8`, `fp16`, `dynamic_int4_block32`,
`dynamic_int4_block128` — not the `ai-edge-quantizer` recipe names, and not a recipe
JSON. Per-layer mixed precision is unavailable here.

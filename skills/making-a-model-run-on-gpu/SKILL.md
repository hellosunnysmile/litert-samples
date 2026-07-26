---
name: making-a-model-run-on-gpu
description: >-
  Diagnoses and fixes a LiteRT-LM model that won't load on the GPU, or that runs but
  slower than expected. Use when engine creation fails, when the delegate reports
  unsupported operations, or when deciding which artifact to ship per accelerator.
  Don't use for quantization quality problems — see choosing-quantization.
---

# Making a model run on the GPU

## The failure that looks like a crash but is a design constraint

```
ERROR: Following operations are not supported by GPU delegate:
GATHER_ND: Operation is not supported.
1236 operations will run on the GPU, and the remaining 52 operations will run on the CPU.
ERROR: Hint fully delegated to single delegate is set, but the graph is not fully delegated.
E0000 engine.cc:807] Failed to create engine: INTERNAL
```

**LiteRT-LM requires the delegate to take the graph whole.** One unsupported op out of
1288 costs the entire accelerator — there is no partial-delegation fallback. So "will it
run on the GPU" is a property of the *graph*, decided at export time, not a runtime flag.

## The fix that worked

The `GATHER_ND` above comes from the causal mask being **looked up inside the graph**.
Hand the mask in at run time instead and it disappears:

```bash
etf produce <model> --author <arch> --set mask_as_input=true
```

Measured on two models, same weights, same int8 quantization, four graphs:

| Build of SmolLM2-135M | GPU | CPU |
| :-- | --: | --: |
| traced (default export) | **166.1 tok/s** | 122.0 tok/s |
| re-authored, default | ❌ won't load | 119.6 tok/s |
| re-authored, transposed KV cache | ❌ won't load | **141.5 tok/s** |
| re-authored, mask as input | 96.3 tok/s | 122.1 tok/s |

Qwen3-0.6B reproduces the same shape (traced 90.2 GPU; transposed-KV 46.9 CPU vs traced
36.7; transposed won't load on GPU).

Three conclusions:

1. `mask_as_input` is what makes a re-authored graph GPU-eligible. Nothing else in the
   sweep changed delegation.
2. **The best CPU build is not the best GPU build.** A transposed KV cache is the fastest
   thing measured on CPU (+16% and +28% on two models) and keeps the model off the GPU
   entirely. Ship both artifacts and let the router choose per device.
3. For a model the tracer handles, **tracing wins the GPU** by 32–42%. Re-authoring is
   the escape hatch and a lever on graph shape — not a free speedup.

## Record it, don't rediscover it

A failed engine creation is a fact about the artifact, not a flaky run. Write it into
the manifest so routing stops offering an artifact the runtime already refused:

```bash
etf bench <model-id> --variant <name> --accelerator gpu
#   failed on gpu: Failed to create LiteRT-LM engine for …
#   recorded: …litertlm can't run on gpu (~/.etf/manifests/<model>.json)
```

`runnable_on` loses that accelerator and `etf run` picks a build that works.

## GPU knobs that did nothing (so don't spend time on them)

On Qwen3-0.6B, against a 90.1 tok/s baseline:

- `enable_gpu_dynamic_prefill` / `enable_gpu_dynamic_cache` (round shapes to the
  "magic numbers" the GPU runtime prefers): 89.7 tok/s.
- `externalize_embedder`: 88.4 tok/s, and a *larger* artifact (771 MB vs 613 MB).
- `experimental_use_fp16`: doesn't trace at all — dies in `bmm` with
  `expected scalar type torch.float32 but found torch.float16`.

Where the GPU time actually goes is weight bandwidth. See choosing-quantization.

## Cold start is a separate cost

LiteRT's GPU runtime compiles kernels and repacks weights on first load, then caches
both **next to the artifact** (`*_mldrift_program_cache.bin`, `*_mldrift_weight_cache.bin`
— the weight cache is roughly model-sized).

```bash
etf bench <model-id> --cold    # clears those caches first
etf bench <model-id>           # every launch after the first
```

Measured on a 585 MB artifact: 1.1 s cold vs 0.8 s warm, reproducible across pairs.
Decode is unaffected — the cache only changes what happens before the first token. If
first-launch latency matters, ship the cache or warm it on install.

---
name: measuring-model-changes
description: >-
  Measures whether an on-device model change actually helped, without fooling yourself.
  Use when comparing quantizations, graph configurations, accelerators or runtimes,
  when a benchmark result looks too good, or before publishing performance numbers.
  Don't use for setting up conversion — see converting-hf-llm-to-litertlm.
---

# Measuring a model change honestly

Every optimization in this stack is a trade, so a number without its counterpart is
noise dressed as progress. These are the rules that survived contact with real runs.

## 1. Make the evaluation reproducible before you compare anything

Running one artifact through one suite three times gave **11/11, 9/11, 9/11**. Not a
regression — the `.litertlm` carries the source model's sampler settings (Qwen3 defaults
to temperature 0.6, top-p 0.95), so the eval was sampling.

A gate that swings ±2/11 on its own cannot rank variants. Decode **greedily**:

```python
conversation = engine.create_conversation(
    sampler_config=litert_lm.SamplerConfig(top_k=1)   # top-1, no sampling
)
```

ETF does this whenever `temperature == 0` (its default). After the change, two runs
produced identical scores, identical failures, and identical answer text. Only then are
differences meaningful — and on an 11-case suite, a one-case difference still isn't.

## 2. Measure speed and quality together, per artifact

The fastest build of a model is regularly its worst. On Qwen3-0.6B, channelwise int4 was
the fastest artifact produced (113.3 tok/s) and answered 2 of 11 questions.

So: record the pass rate **against the artifact**, not the model, and refuse to route a
variant measured below the gate. ETF writes `cost.pass_rate` into the variant and the
router applies it:

```
[route] Qwen3-0.6B-produced -> quant=mixed_int4_down8 measured=95.7tok/s
        (below gate: dynamic_int4 18%, dynamic_int4_b32 45%, …)
```

A variant nobody has evaluated stays routable — otherwise a freshly produced model can
never run — but one that has been measured and failed does not come back.

## 3. Hold the hardware fixed

An early comparison ran MLX on the GPU against LiteRT-LM on the CPU and implied a 3.0×
win. Both on the GPU, it was 1.6×. Sweep instead of assuming:

```bash
etf bench <model-id> --variants --accelerator gpu   # one accelerator, every variant
etf bench <model-id> --matrix                       # every accelerator a variant runs on
```

## 4. Measure the four numbers that describe different costs

| Metric | What it is | Why separately |
| :-- | :-- | :-- |
| `decode_tps` | steady-state tokens/s | what long generations feel like |
| `ttft_ms` | time to first token | what short agentic turns feel like |
| `load_ms` | weights + kernel compilation | what launch feels like; `--cold` clears the GPU cache |
| `pass_rate` | capability suite score | what all the others cost you |

The ranking flips between them. Across four frameworks on one Qwen3-0.6B, MLX led
throughput (216 tok/s), llama.cpp led latency by 3.5–4× (42 ms TTFT), and no framework
won outright. Report the metric that matches the workload, and say which one you chose.

Never mix `avg_tps` (end-to-end, includes prefill) with `decode_tps` — runtimes that
can't stream only support the former, and it is not comparable.

## 5. Watch for measurement that changes the measurement

- The resident engine cache keeps every model it loaded; benchmarking six variants left
  ~7 GB resident and made later memory numbers meaningless. Release engines between
  candidates (`release_engines()` in ETF's resident backend).
- Model memory is only attributable for in-process runtimes. A model served by Ollama or
  a cloud endpoint lives in another process — report "n/a", never your own RSS.
- The first GPU run of an artifact pays for kernel compilation. Warm up, or measure cold
  deliberately, but don't mix the two in one table.

## 6. Say which machine a number describes

`darwin/arm64` is not a machine. An M1 Air and an M4 Max both report it and differ
several-fold in GPU throughput, so quoting one at the other is how a shared performance
database becomes misleading rather than useful.

Keep two things apart:

- A **fingerprint** — chip, CPU/GPU core counts, RAM, OS version — recorded with every
  measurement, so two numbers can be known to be comparable at all.
- A **platform class** — platform × accelerator × framework, e.g. "macOS arm64 · GPU ·
  litert-lm". This is what a *published* recommendation is for. Optimizing for the class
  is the useful thing to publish; nobody can pre-optimize for every machine.

Then, when answering "what should I run here", fall back one tier at a time and **name
the tier you used**:

```bash
etf perf --reference <model-id>
# This machine: Apple M1 Pro, 16-core GPU, 32 GB, darwin 26.5.2
# Best decode_tps (higher is better), measured on **this machine**:
#      196.6  Gemma3-270M-produced  gpu  …_mixed_int4_down8.litertlm  (Apple M1 Pro)
```

this machine → the same chip → the same chip family → the same platform class. Below
that floor, the honest answer is "benchmark it locally", not a number from unrelated
hardware. `--min-tier` sets the floor.

## 7. Keep the history, not just the verdict

Speed and quality both go in one perf database (`~/.etf/perf.db`), keyed by model,
variant, accelerator, engine and device. That is what makes "did this get worse?"
answerable at all:

```bash
etf perf --compare                      # best decode per variant/accelerator
etf perf --compare --metric ttft_ms     # ...or by latency
etf perf <model-id> --kind eval         # quality history
etf perf --export perf-snapshot.json    # a publishable, reviewable snapshot
```

Export JSON, not the SQLite file: a database is a binary blob that conflicts on every
merge, while a schema-versioned JSON snapshot diffs, reviews and merges — and it is an
append-only record of facts, which is what measurements are.

A measurement you didn't record is a measurement you will take again.

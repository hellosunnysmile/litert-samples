---
name: choosing-quantization
description: >-
  Chooses and verifies a quantization recipe for an on-device LLM, including per-layer
  mixed precision that beats uniform int8 and int4. Use when a converted model is too
  big or too slow, when int4 wrecked its accuracy, or when writing an ai-edge-quantizer
  recipe. Don't use for graph or accelerator problems — see making-a-model-run-on-gpu.
---

# Choosing a quantization recipe

Decode on a GPU is bandwidth-bound: every token reads the whole weight set, so weight
format is the largest single lever you have. It is also the easiest place to destroy a
model without noticing.

## The result to start from

Qwen3-0.6B, one float model quantized every way, measured on an M-series GPU with
**greedy** decoding and an 11-case suite. (Two more models follow — the conclusion
changes with model size, so read both sections before choosing.)

| Build | Size | Decode | Quality |
| :-- | --: | --: | --: |
| all int8 | 585 MB | 90.2 tok/s | 9/11 |
| **int4 + int8 down-projections** | **363 MB** | **95.7 tok/s** | **10/11** |
| int4 + int8 attention projections | 402 MB | 92.0 tok/s | 9/11 |
| int4 + int8 first/last blocks | 353 MB | 95.5 tok/s | 6/11 |
| all int4, blockwise-32 | 325 MB | 101.6 tok/s | 5/11 |
| all int4, channelwise | 301 MB | 113.3 tok/s | 2/11 |

Read three things out of it:

1. **Uniform int4 is a cliff, not a trade.** Channelwise int4 buys 26% decode and loses
   7 of 11 answers. Blockwise-32 (one scale per 32 weights instead of per channel) gets
   half of that quality back and gives up half the speed.
2. **One mixed policy dominates int8** — 38% smaller, 6% faster, no worse quality.
3. **It's the down-projections that can't take 4 bits.** Protecting 27 of them beat
   protecting 110 attention projections, which cost 2.5× the bytes for nothing.
   Position in the stack (first/last blocks) didn't matter at all.

## Model size decides how much quantization a model can take

Three models through the same six builds, same machine, greedy decoding, 11-case suite.
Read the **quality** column down the table, not across:

| Build | Qwen3-0.6B | Gemma 3 270M | Gemma 3 1B |
| :-- | --: | --: | --: |
| all int8 | 9/11 | **8/11** | 9/11 |
| int4 + int8 down-projections | **10/11** | 7/11 | **10/11** |
| int4 + int8 attention projections | 9/11 | 7/11 | 9/11 |
| int4 + int8 first/last blocks | 6/11 | 3/11 | 9/11 |
| all int4, blockwise-32 | 5/11 | 4/11 | 8/11 |
| int4 + int8 embedding | — | 4/11 | 9/11 |

**Bigger models absorb quantization; small ones don't.** At 1B, uniform int4 costs one
case and every mixed build matches int8 while being 35–41% smaller and 26–34% faster —
int8 is dominated outright. At 0.6B and below, uniform int4 is a cliff (9/11 → 2/11 on
Qwen3), and on the 270M model *nothing* int4-based clears a 0.7 gate at all.

So the starting point depends on the size you're working with:

| Model size | Start with | Because |
| :-- | :-- | :-- |
| ≥ 1B | `attn8` or `down8`, and try uniform int4 | int8 is likely dominated; the risk is low |
| ~0.5–1B | `down8`, and check it against int8 | mixed usually wins, uniform int4 usually doesn't |
| < 500M | int8, and treat every int4 build as a candidate to disprove | the model may have no headroom at all |

Three things held on all three models:

- **`down8` never lost.** It scored at or above every other policy every time (10/11,
  7/11, 10/11), even where it wasn't the fastest.
- **Protecting the embedding is a bad trade.** Largest artifact of the six on every
  model, and never the best quality.
- **The regexes generalise.** Llama-style and Gemma-style module names differ, but the
  `nn.Linear` attribute names don't, and the dtype readback confirms the matches.

One thing that stopped holding at 1B: **size stopped predicting speed.** The 1128 MB
`embed8` build decodes at 87.8 tok/s while the 546 MB uniform-int4 build manages 77.3 —
int4 unpacking on the embedding lookup costs more than the bytes it saves. Don't infer
throughput from a file listing; measure it.

## What transferred to a second model, and what didn't

The same six builds on Gemma 3 270M (GPU, greedy, same suite):

| Build | Size | Decode | Quality |
| :-- | --: | --: | --: |
| all int8 | 272 MB | 164.6 tok/s | **8/11** |
| int4 + int8 down-projections | 162 MB | 189.1 tok/s | 7/11 |
| int4 + int8 attention projections | 165 MB | 188.0 tok/s | 7/11 |
| int4 + int8 first/last blocks | 162 MB | 187.2 tok/s | 3/11 |
| all int4, blockwise-32 | 152 MB | 189.4 tok/s | 4/11 |
| int4 + int8 embedding | 478 MB | 186.0 tok/s | 4/11 |

**The winner flipped.** On Qwen3 the mixed build dominated int8 outright; here it is 40%
smaller and 15% faster but one case worse, so against a 0.7 gate the int8 build is what
gets served. Same policies, same machine, opposite verdicts.

What did transfer, on both:

- **Protecting projection matmuls works** — `down8` and `attn8` recover most of what
  uniform int4 loses (4/11 → 7/11 here, 5/11 → 10/11 on Qwen3).
- **Protecting by position, or protecting the embedding, does not.** `edges8` was the
  worst build on both models. `embed8` produced the *largest* artifact of the six and no
  quality benefit at all.
- **The regexes generalise.** Gemma 3 names its modules `Gemma3DecoderLayer` /
  `Gemma3MLP`, but the `nn.Linear` attribute names are shared, and the dtype readback
  confirms the matches.

So: **start with `down8`, and always re-run the comparison.** The policy that wins is
model-dependent; the policies that lose appear to be universal. Notice too that Gemma 3
270M is embedding-dominated (80 MB of a 133 MB artifact even at 4 bits), which is why
matmul precision moves so little of its speed — check that shape before predicting what
a quantization will buy.

## Writing the recipe

An `ai-edge-quantizer` recipe is a list of rules: a regex over an op's **scope name**,
plus the precision to use where it matches. This works because a converted LLM keeps the
PyTorch module path in its tensor names:

```
…/LlamaDecoderLayer_23/LlamaAttention_self_attn/torch.nn.modules.linear.Linear_q_proj;
…/LlamaDecoderLayer_15/LlamaMLP_mlp/torch.nn.modules.linear.Linear_down_proj;
```

`q_proj`, `down_proj`, `self_attn` are HuggingFace attribute names shared across Llama,
Qwen, Mistral and Phi, so one regex addresses all of them.

```bash
python utilities/make_quant_recipe.py down8 > recipe.json
etf produce Qwen/Qwen3-0.6B --quantize ./recipe.json
# or: etf produce Qwen/Qwen3-0.6B --quantize mixed_int4_down8
```

Four rules for writing one by hand:

- **Bulk rule first, overrides after.** Rules apply in order and the *last* match wins.
- **The bulk rule must use `"operation": "*"`.** Naming `FULLY_CONNECTED` leaves every
  other quantizable op in float, and the artifact comes out *larger* than an int8 build.
- **4-bit needs `BLOCKWISE_32`; 8-bit needs `CHANNELWISE`.** The quantizer rejects 8-bit
  blockwise outright, so a policy that gets this wrong silently fails to convert.
- **Don't protect the embedding table** on a large vocabulary. With all 191 matmuls at
  int4, lifting only the embedding to int8 produced a 624 MB artifact — bigger than
  quantizing the entire model to int8. The size arithmetic doesn't explain that on its
  own (the embedding is 148 MB of a 555 MB int8 build), but it reproduces, so measure
  before you protect it.

## Verify that the recipe did what it says

This is not optional. A recipe that matches nothing produces a model that looks fine and
is unchanged, and a recipe that matches too much produces one that is silently ruined.

```bash
~/.etf/produce-venv/bin/python utilities/inspect_quantization.py model.tflite
#   FULLY_CONNECTED   INT4   164 tensors  163.0 MB
#   FULLY_CONNECTED   INT8    27 tensors   81.0 MB
#   EMBEDDING_LOOKUP  INT4     1 tensors   74.2 MB

python utilities/inspect_quantization.py model.tflite --match Linear_down_proj
```

If the counts don't match the policy, the recipe is wrong — fix it before measuring
anything else. Both of the size anomalies above were found this way and by nothing else.

## Then measure both axes, not one

```bash
etf produce <model> --sweep int4      # or --sweep mixed
etf bench <model-id> --variants --accelerator gpu
etf eval  <model-id> --variant <name>
```

Take the **fastest build that stays inside your quality budget**, never the fastest
build. And record the pass rate against the artifact, not just the model: ETF writes
`cost.pass_rate` into the variant so the router refuses a fast build that can't answer.

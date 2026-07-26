# Gemma 3 270M → LiteRT-LM

Recipes and measurements for converting Gemma 3 270M to a `.litertlm`, and — more
usefully — the second data point on whether a quantization policy transfers between
model families.

## Measured

Six builds from one checkpoint, M-series laptop GPU, greedy decoding, 11-case suite:

| Build | Size | Decode | Quality |
| :-- | --: | --: | --: |
| all int8 | 272 MB | 164.6 tok/s | **8/11** |
| int4 + int8 down-projections | **162 MB** | **189.1 tok/s** | 7/11 |
| int4 + int8 attention projections | 165 MB | 188.0 tok/s | 7/11 |
| int4 + int8 first/last blocks | 162 MB | 187.2 tok/s | 3/11 |
| all int4, blockwise-32 | 152 MB | 189.4 tok/s | 4/11 |
| int4 + int8 embedding | 478 MB | 186.0 tok/s | 4/11 |

**The verdict here is the opposite of Qwen3-0.6B's**, and that is the point. On Qwen3 the
mixed `down8` build dominated int8 outright (smaller, faster, no less accurate). Here it
is 40% smaller and 15% faster but scores 7/11 against int8's 8/11 — so against a 0.7
gate, `etf run` serves the **int8** build for this model and the mixed build for Qwen3.
Same policies, same machine, different answer, decided by measurement rather than by a
rule of thumb.

Two things that did transfer, on both models:

- **Protecting projection matmuls works.** `down8` and `attn8` both recover most of what
  uniform int4 loses (4/11 → 7/11 here, 5/11 → 10/11 on Qwen3).
- **Protecting by position or protecting the embedding does not.** `edges8` is the worst
  build on both models; `embed8` gives the *largest* artifact and no quality benefit.

Why the mixed builds barely differ in speed here: Gemma 3 270M is embedding-dominated —
80 MB of a 133 MB artifact is the token embedding even at 4 bits — so matmul precision
moves much less weight traffic than it does on Qwen3.

## Reproduce

```bash
etf produce google/gemma-3-270m-it --cache-length 1024 --sweep mixed
etf bench Gemma3-270M-produced --variants --accelerator gpu
etf eval  Gemma3-270M-produced --variant mixed_int4_down8
```

`google/gemma-3-270m-it` is gated: accept the licence on the model page and run
`huggingface-cli login` once. The numbers above were produced from the open mirror
`unsloth/gemma-3-270m-it`, which is the same weights — re-run against the official
checkpoint before publishing an artifact.

Verify the recipe matched before trusting any of it — Gemma 3 names its modules
`Gemma3DecoderLayer` / `Gemma3MLP`, and the regexes are written against the `nn.Linear`
attribute names, which are shared:

```bash
python ../../../../utilities/inspect_quantization.py <artifact>.litertlm
#   EMBEDDING_LOOKUP  INT4    1 tensors   80.0 MB
#   FULLY_CONNECTED   INT4   68 tensors   31.9 MB
#   FULLY_CONNECTED   INT8   17 tensors   21.2 MB   <- the down-projections matched
```

## Files

| File | What it is |
| :-- | :-- |
| `gemma3_270m_int4_down8.quant.json` | 4-bit bulk, 8-bit `Linear_down_proj`. Model-independent — the same file works for any HuggingFace decoder LM. |
| `gemma3_270m_int4_down8.recipe.json` | The production receipt. `etf produce <this file>` replays it. |

## Not done

Gemma 3 1B and 4B, and the vision-capable variants. The 270M model's embedding-dominated
shape makes it a poor guide for the larger ones.

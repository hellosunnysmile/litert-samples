# Gemma 3 → LiteRT-LM

Recipes and measurements for converting Gemma 3 to a `.litertlm`, at two sizes — and,
more usefully, the evidence on whether a quantization policy transfers between model
families and between sizes. It does not transfer blindly, and the two sizes here
disagree with each other.

## Gemma 3 270M — where int4 hurts

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

## Gemma 3 1B — where quantization stops being scary

Same six builds, same machine and suite, from `google/gemma-3-1b-it`:

| Build | Size | Decode | Quality |
| :-- | --: | --: | --: |
| all int8 | 979 MB | 66.2 tok/s | 9/11 |
| int4 + int8 down-projections | 633 MB | 83.4 tok/s | **10/11** |
| int4 + int8 attention projections | **579 MB** | **88.4 tok/s** | 9/11 |
| int4 + int8 first/last blocks | 592 MB | 88.8 tok/s | 9/11 |
| all int4, blockwise-32 | 546 MB | 77.3 tok/s | 8/11 |
| int4 + int8 embedding | 1128 MB | 87.8 tok/s | 9/11 |

**Uniform int8 is dominated outright here** — every mixed build is 35–41% smaller and
26–34% faster, and none of them is less accurate. At 1B the model is robust enough that
even *uniform* int4 only drops one case (9/11 → 8/11), which is the opposite of the 270M
model and of Qwen3-0.6B, where uniform int4 was a cliff.

Practical picks: `attn8` for the best speed/size (579 MB, 88.4 tok/s, 9/11), `down8` if
you want the highest measured quality (633 MB, 83.4 tok/s, 10/11).

Two things worth noticing:

- **Size stopped predicting speed.** The 1128 MB `embed8` build decodes at 87.8 tok/s
  while the 546 MB uniform-int4 build manages 77.3. Whatever int4 costs to unpack on the
  embedding lookup outweighs the bytes it saves, on this runtime.
- **`down8` has now scored at or above every other policy on all three models** — Qwen3
  10/11, Gemma3 270M 7/11, Gemma3 1B 10/11 — even where it isn't the fastest.

## Reproduce

```bash
etf produce google/gemma-3-1b-it   --cache-length 1024 --sweep mixed
etf produce google/gemma-3-270m-it --cache-length 1024 --sweep mixed
etf bench Gemma3-1B --variants --accelerator gpu
etf eval  Gemma3-1B --variant mixed_int4_down8
```

Both repos are gated: accept the licence on the model page with your HuggingFace account
and run `huggingface-cli login` once. The 1B numbers come from the official checkpoint;
the 270M numbers were produced from the open mirror `unsloth/gemma-3-270m-it` (same
weights) before access was granted.

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
| `gemma3_270m_int4_down8.recipe.json` | The 270M production receipt. `etf produce <this file>` replays it. |
| `gemma3_1b_int4_down8.recipe.json` | The 1B production receipt. |

## Not done

Gemma 3 4B and above, and the vision-capable variants (the exporter has an
`image_text_to_text` task that nothing here has exercised).

# Lightweight `klein_bins` — reproduce on a 32 GB machine

The upstream flow (`conversion/gen_prep_klein.py`) stages the 13 host tensors the
Android app reads by running the **full fp32 `Flux2KleinPipeline` once**. That
needs ~24 GB of weights downloaded and, per `conversion/README.md`, **~40 GB of
RAM** — more than a typical 32 GB laptop. It also isn't strictly necessary: the
app's own `PromptEncoder.kt` notes that every staged tensor depends only on the
**prompt, positions, the schedule, or the seed** — never on the 4B forward pass.

`gen_bins_light.py` uses that to produce the same `klein_bins` with a **~20 MB
download and no 40 GB pipeline**:

| tensor(s) | how it's produced here |
|---|---|
| `cos`, `sin`, `enc_cos`, `enc_sin`, `latents0`, `dsigma`, `unpack_perm`, `unpatch_perm` | real diffusers pipeline methods, driven **config-only** (no weights) |
| `temb` | `transformer.time_guidance_embed` — its ~20 MB of weights are pulled from the 7.75 GB transformer safetensors via **HTTP range requests**; the 4B blocks are instantiated on `meta` and never materialized |
| `bn_mean`, `bn_std` | VAE batch-norm buffers, range-fetched (a few KB) |
| `inputs_embeds`, `enc_mask` | Qwen tokenizer + the published fp16 embedding table |

Nothing here changes the numbers the graphs see — it only avoids the fp32 forward
that the upstream script runs solely to (a) get a `ref_fp32.png` for PSNR and
(b) capture tensors that are actually weight-independent.

## Usage

```bash
pip install torch "git+https://github.com/huggingface/diffusers" transformers \
            accelerate safetensors ml_dtypes pillow numpy requests huggingface_hub

# 1. get the 12 int8 graphs (and optional tokenizer/ for editable prompts)
#    from https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT  ->  <graphs_dir>

# 2. produce klein_bins (downloads ~1 MB configs + ~20 MB weights + the 778 MB
#    embedding table; pass --embed <graphs_dir>/tokenizer/qwen_embed_fp16.bin to reuse it)
python gen_bins_light.py --out <graphs_dir>/klein_bins \
       --prompt "a red apple on a wooden table, studio lighting" \
       --embed <graphs_dir>/tokenizer/qwen_embed_fp16.bin

# 3. (optional) sanity-check on the host CPU — needs ai-edge-litert
pip install ai-edge-litert
python host_verify_light.py --graphs <graphs_dir> \
       --bins <graphs_dir>/klein_bins --out check.png

# 4. build + stage + run, exactly as the sample README says
cd ../../android
./gradlew :app:installDebug
./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins
```

If `<graphs_dir>/tokenizer/` is staged, the app becomes prompt-editable and
recomputes `inputs_embeds`/`enc_mask` on device; the bins here cover the other 11
(prompt-independent) tensors plus a default baked prompt.

## Verified

`host_verify_light.py` drives the real 12 int8 graphs with these bins and renders
the prompt correctly (validated against the fp32 sample output). The same bins,
staged with `install_to_device.sh`, generate on device with the graphs reporting
`Replacing N out of N node(s) with delegate (LITERT_CL)` — 100% on the GPU, as the
sample README describes. Measured end-to-end ≈112–120 s on a Snapdragon 8 Elite
(Adreno 830), vs. ~306 s on the Pixel 8a in the sample README.

## Limitations

- No `ref_fp32.png`, so no PSNR gate — this path is for *running* the model, not
  for reproducing the accuracy numbers. Use the upstream `gen_prep_klein.py` on a
  high-RAM machine for that.
- Text-to-image only. Editing (`kce_*` graphs, `patch_perm`, `image_latents`) is
  not covered here.
- Relies on `Flux2KleinPipeline` from the diffusers **main** branch (not yet in a
  released version) and a transformers new enough to include Qwen3.

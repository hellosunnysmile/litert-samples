"""Produce the FLUX.2-klein `klein_bins` WITHOUT the 40 GB fp32 pipeline.

`conversion/gen_prep_klein.py` stages the 13 host tensors the Android app reads
by running the full fp32 `Flux2KleinPipeline` once. That needs ~24 GB of weights
downloaded and ~40 GB of RAM -- more than a typical 32 GB laptop has.

This script produces byte-identical-in-shape, numerically-equivalent bins on a
32 GB machine with a tiny download, by exploiting a fact the app itself relies on
(see `PromptEncoder.kt`): every staged tensor depends only on the prompt, the
positions, the schedule, or the seed -- never on the 4B transformer / text-encoder
*forward pass*. So:

  * geometry (cos/sin, enc rotary, latents0, dsigma, unpack/unpatch perms)
        -> real diffusers pipeline methods, driven config-only (no weights)
  * temb  -> transformer.time_guidance_embed, whose ~20 MB of weights are pulled
             from the 7.75 GB transformer safetensors with HTTP range requests;
             the 4B blocks are instantiated on `meta` and never materialized
  * bn    -> the VAE batch-norm buffers, range-fetched (a few KB)
  * inputs_embeds / enc_mask -> Qwen tokenizer + the published fp16 embedding table

Total download is only config JSONs (~1 MB) + ~20 MB of weights + the 778 MB
embedding table (skip with --embed if you already have it from the LiteRT repo).

Usage:
    pip install torch "git+https://github.com/huggingface/diffusers" transformers \
                accelerate safetensors ml_dtypes pillow numpy requests huggingface_hub
    python gen_bins_light.py --out klein_bins --prompt "a red apple on a wooden table"

Then stage as usual:
    cd ../../android
    ./gradlew :app:installDebug
    ./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins

`<graphs_dir>` holds the 12 .tflite graphs from `litert-community/FLUX.2-klein-4B-LiteRT`
(and its `tokenizer/` folder if you want editable prompts). This script writes the
`klein_bins/` those graphs consume.
"""
import argparse
import json
import os
import struct
import time

import ml_dtypes
import numpy as np
import requests
import torch

REPO = "black-forest-labs/FLUX.2-klein-4B"          # fp32 config source (Apache-2.0)
LITERT_REPO = "litert-community/FLUX.2-klein-4B-LiteRT"  # published fp16 embed table
STEPS, SIZE, SEED = 4, 256, 1234
SEQ_TXT, SEQ_IMG, ENC_HEADS, NEG = 512, 256, 32, -1e9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="klein_bins", help="output bins directory")
    p.add_argument("--prompt", default="a red apple on a wooden table, studio lighting")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cache", default=".flux2_cfg_cache", help="dir for downloaded configs")
    p.add_argument("--embed", default=None,
                   help="path to qwen_embed_fp16.bin (else downloaded from the LiteRT repo)")
    p.add_argument("--repo", default=REPO)
    return p.parse_args()


# ---- HTTP range helpers: read individual tensors from a remote safetensors file ----

def rget(url, rng):
    for attempt in range(8):
        try:
            r = requests.get(url, headers={"Range": rng}, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500:
                raise SystemExit(f"HTTP {status} for range {rng}; not retrying (client error)")
            print(f"  [retry {attempt}] HTTP {status}")
            time.sleep(3)
        except requests.RequestException as e:
            print(f"  [retry {attempt}] {type(e).__name__}")
            time.sleep(3)
    raise SystemExit(f"range GET failed: {rng}")


def st_header(url):
    n = struct.unpack("<Q", rget(url, "bytes=0-7"))[0]
    return json.loads(rget(url, f"bytes=8-{8 + n - 1}")), 8 + n


def st_fetch(url, hdr, base, key):
    o = hdr[key]["data_offsets"]
    raw = rget(url, f"bytes={base + o[0]}-{base + o[1] - 1}")
    dt = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "F16": np.float16}[hdr[key]["dtype"]]
    return np.frombuffer(raw, dtype=dt).astype(np.float32).reshape(hdr[key]["shape"]).copy()


def main():
    args = parse_args()
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download, hf_hub_url
    from transformers import AutoTokenizer, AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from diffusers import Flux2Transformer2DModel, Flux2KleinPipeline as P
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    os.makedirs(args.out, exist_ok=True)

    # ---- tiny config files (~1 MB total) ----
    for f in ["transformer/config.json", "scheduler/scheduler_config.json",
              "vae/config.json", "text_encoder/config.json"]:
        hf_hub_download(args.repo, f, local_dir=args.cache)
    from huggingface_hub import snapshot_download
    snapshot_download(args.repo, allow_patterns=["tokenizer/*"], local_dir=args.cache)

    def save(t, name):
        a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        a.astype("<f4").tofile(os.path.join(args.out, f"{name}.bin"))

    def save_int(a, name):
        np.asarray(a, dtype="<i4").tofile(os.path.join(args.out, f"{name}.bin"))

    # ---- transformer: meta (no RAM) + materialize pos_embed + time_guidance_embed ----
    with open(f"{args.cache}/transformer/config.json") as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with init_empty_weights():
        tr = Flux2Transformer2DModel(**cfg)
    tr.pos_embed.to_empty(device="cpu")
    tr.time_guidance_embed.to_empty(device="cpu")
    turl = hf_hub_url(args.repo, "transformer/diffusion_pytorch_model.safetensors")
    thdr, tbase = st_header(turl)
    tsd = {k.replace("time_guidance_embed.", ""): torch.from_numpy(st_fetch(turl, thdr, tbase, k))
           for k in thdr if "time_guidance_embed" in k}
    tr.time_guidance_embed.load_state_dict(tsd, strict=False)
    tr = tr.float().eval()
    print(f"[transformer] range-fetched {list(tsd)} (~20 MB), 4B blocks stayed on meta")

    # ---- geometry from real pipeline methods (config-only stub) ----
    stub = P.__new__(P)
    stub.vae_scale_factor = 8
    g = torch.Generator("cpu").manual_seed(args.seed)
    latents, latent_ids = P.prepare_latents(stub, 1, cfg["in_channels"] // 4,
                                            SIZE, SIZE, torch.float32, "cpu", g)
    img_ids = latent_ids[0]
    text_ids = P._prepare_text_ids(torch.zeros(1, SEQ_TXT, 8))[0]
    with torch.no_grad():
        ic, isn = tr.pos_embed(img_ids)
        tc, tsn = tr.pos_embed(text_ids)
    cos = torch.cat([tc, ic], 0)[:, 0::2][None, :, None, :]
    sin = torch.cat([tsn, isn], 0)[:, 0::2][None, :, None, :]

    # ---- schedule + temb  (pipeline feeds transformer `t/1000`, gen_prep restores `t`) ----
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{args.cache}/scheduler")
    mu = compute_empirical_mu(latents.shape[1], STEPS)
    ts, _ = retrieve_timesteps(sch, STEPS, "cpu", sigmas=None, mu=mu)
    sig = sch.sigmas.numpy()
    dsigma = sig[1:STEPS + 1] - sig[:STEPS]
    with torch.no_grad():
        temb = torch.cat([tr.time_guidance_embed(t.view(1).float(), None) for t in ts], 0)
    print(f"[sched] timesteps {[round(float(t), 2) for t in ts]}  "
          f"dsigma {[round(float(d), 4) for d in dsigma]}")

    # ---- tail permutations (pure gathers) ----
    unpack = P._unpack_latents_with_ids(torch.arange(SEQ_IMG * 128).float().reshape(1, SEQ_IMG, 128),
                                        img_ids[None], 16, 16).flatten().round().long().numpy()
    unpatch = P._unpatchify_latents(torch.arange(128 * 16 * 16).float().reshape(1, 128, 16, 16)
                                    ).flatten().round().long().numpy()

    # ---- VAE batch-norm (range-fetched) ----
    vurl = hf_hub_url(args.repo, "vae/diffusion_pytorch_model.safetensors")
    vhdr, vbase = st_header(vurl)
    bn_mean = torch.from_numpy(st_fetch(vurl, vhdr, vbase, "bn.running_mean"))
    bn_var = torch.from_numpy(st_fetch(vurl, vhdr, vbase, "bn.running_var"))
    with open(f"{args.cache}/vae/config.json") as f:
        eps = json.load(f).get("batch_norm_eps", 1e-5)
    bn_std = torch.sqrt(bn_var + eps)

    # ---- prompt-dependent tensors: tokenize + embed (fp16 table) + mask + enc rotary ----
    embed_path = args.embed
    if embed_path is None:
        embed_path = hf_hub_download(LITERT_REPO, "tokenizer/qwen_embed_fp16.bin",
                                     local_dir=args.cache)
    qcfg = AutoConfig.from_pretrained(f"{args.cache}/text_encoder")
    tok = AutoTokenizer.from_pretrained(f"{args.cache}/tokenizer")
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=SEQ_TXT)
    ids, am = enc["input_ids"], enc["attention_mask"]
    tbl = np.fromfile(embed_path, dtype=np.float16).reshape(-1, qcfg.hidden_size)
    inputs_embeds = torch.from_numpy(tbl[ids[0].numpy()].astype(np.float32))[None]
    keep = am[0].float()
    mask = (torch.full((SEQ_TXT, SEQ_TXT), NEG).triu(1) + torch.where(keep > 0, 0.0, NEG)[None, :])[None, None]
    mask = mask.expand(1, ENC_HEADS, SEQ_TXT, SEQ_TXT).contiguous()
    enc_cos, enc_sin = Qwen3RotaryEmbedding(qcfg)(inputs_embeds, torch.arange(SEQ_TXT)[None])
    print(f"[text] {int(keep.sum())}/{SEQ_TXT} real tokens for prompt: {args.prompt!r}")

    # ---- write the 13 bins the app reads (see Flux2KleinGenerator.kt) ----
    save(inputs_embeds, "inputs_embeds")
    save(mask, "enc_mask")
    save(enc_cos, "enc_cos")
    save(enc_sin, "enc_sin")
    save(cos, "cos")
    save(sin, "sin")
    save(temb, "temb")
    save(dsigma, "dsigma")
    save(latents, "latents0")
    save(bn_mean, "bn_mean")
    save(bn_std, "bn_std")
    save_int(unpack, "unpack_perm")
    save_int(unpatch, "unpatch_perm")
    print(f"[done] wrote 13 bins -> {args.out}/")


if __name__ == "__main__":
    main()

"""Run the real 12-graph FLUX.2-klein loop on host CPU with lightweight bins.

Like `conversion/gen_verify_klein.py`, but points at explicit graphs/bins dirs
and skips the PSNR-vs-fp32 gate (the lightweight flow has no `ref_fp32.png`).
If the saved image looks like the prompt, the `klein_bins` are correct and will
work on the phone.

    pip install ai-edge-litert torch pillow numpy
    python host_verify_light.py --graphs <graphs_dir> --bins <graphs_dir>/klein_bins --out out.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from ai_edge_litert.compiled_model import CompiledModel

STEPS, SEQ_TXT, SEQ_IMG = 4, 512, 256
DIM_ENC, TAPS = 2560, 3
LATENT_CH, LATENT_HW, PACKED_CH, PACKED_HW, IMAGE_HW = 32, 32, 128, 16, 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", required=True, help="dir with the 12 .tflite graphs")
    p.add_argument("--bins", required=True, help="klein_bins dir from gen_bins_light.py")
    p.add_argument("--out", default="host_loop.png")
    return p.parse_args()


def main():
    args = parse_args()

    def load(n, s):
        return torch.from_numpy(np.fromfile(f"{args.bins}/{n}.bin", "<f4").reshape(s).copy())

    def load_int(n):
        return torch.from_numpy(np.fromfile(f"{args.bins}/{n}.bin", "<i4").astype(np.int64))

    def tfl(name, *inputs):
        m = CompiledModel.from_file(f"{args.graphs}/{name}.tflite")
        sig = m.get_signature_list()
        key = list(sig)[0]
        ind, outd = m.get_input_tensor_details(key), m.get_output_tensor_details(key)
        ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)
        for tn, b, v in zip(sig[key]["inputs"], ib, inputs):
            b.write(np.ascontiguousarray(v, dtype=np.dtype(ind[tn]["dtype"])))
        m.run_by_index(0, ib, ob)
        outs = []
        for tn, b in zip(sig[key]["outputs"], ob):
            d = outd[tn]
            flat = b.read(int(np.prod(d["shape"])), np.dtype(d["dtype"]))
            outs.append(torch.from_numpy(flat.reshape(d["shape"]).copy()))
        del m
        return outs

    inputs_embeds = load("inputs_embeds", (1, SEQ_TXT, DIM_ENC))
    mask = load("enc_mask", (1, 32, SEQ_TXT, SEQ_TXT))
    enc_cos, enc_sin = load("enc_cos", (1, SEQ_TXT, -1)), load("enc_sin", (1, SEQ_TXT, -1))
    cos = load("cos", (1, SEQ_TXT + SEQ_IMG, 1, 64))
    sin = load("sin", (1, SEQ_TXT + SEQ_IMG, 1, 64))
    temb = load("temb", (STEPS, -1))
    dsigma = load("dsigma", (STEPS,))
    latents = load("latents0", (1, SEQ_IMG, PACKED_CH))
    bn_mean, bn_std = load("bn_mean", (-1,)), load("bn_std", (-1,))
    unpack, unpatch = load_int("unpack_perm"), load_int("unpatch_perm")

    print("[enc] 3 encoder chunks ...")
    taps, h = [], inputs_embeds
    for i in range(TAPS):
        h = tfl(f"ke_enc{i}", h.numpy(), mask.numpy(), enc_cos.numpy(), enc_sin.numpy())[0]
        taps.append(h)
    prompt = torch.stack(taps, 1).permute(0, 2, 1, 3).reshape(1, SEQ_TXT, TAPS * DIM_ENC)

    for s in range(STEPS):
        hid, encd, mi, mt, ms = tfl("kc_prep", latents.numpy(), prompt.numpy(), temb[s:s + 1].numpy())
        for i in range(2):
            hid, encd = tfl(f"kc_double{i}", hid.numpy(), encd.numpy(),
                            cos.numpy(), sin.numpy(), mi.numpy(), mt.numpy())
        joint = torch.cat([encd, hid], 1)
        for i in range(4):
            joint = tfl(f"kc_single{i}", joint.numpy(), cos.numpy(), sin.numpy(), ms.numpy())[0]
        noise = tfl("kc_final", joint.numpy(), temb[s:s + 1].numpy())[0][:, :SEQ_IMG]
        latents = latents + dsigma[s] * noise
        print(f"[step {s}] |noise| {noise.norm():.2f}  |latents| {latents.norm():.2f}")

    up = latents.flatten()[unpack].reshape(1, PACKED_CH, PACKED_HW, PACKED_HW)
    up = up * bn_std.view(1, -1, 1, 1) + bn_mean.view(1, -1, 1, 1)
    lat = up.flatten()[unpatch].reshape(1, LATENT_CH, LATENT_HW, LATENT_HW)
    img = tfl("kv_vae", lat.numpy())[0]
    px = ((img[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(px).save(args.out)
    print(f"[done] saved {args.out}  mean_rgb={px.reshape(-1, 3).mean(0).round(1)}")


if __name__ == "__main__":
    main()

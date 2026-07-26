# Utilities

Small tools for the parts of on-device model work that have no good answer in the box.
Each runs standalone; the quantization inspector needs the conversion environment's
interpreter (it reads flatbuffers via `ai-edge-quantizer`).

| Tool | What it answers |
| :-- | :-- |
| [`inspect_quantization.py`](inspect_quantization.py) | What precision does this converted model *actually* use, per layer? Does my regex match anything? |
| [`make_quant_recipe.py`](make_quant_recipe.py) | Give me a mixed-precision recipe for a decoder-only LLM — no dependencies, prints JSON. |

```bash
~/.etf/produce-venv/bin/python utilities/inspect_quantization.py model.tflite
~/.etf/produce-venv/bin/python utilities/inspect_quantization.py model.tflite --match Linear_down_proj
python utilities/make_quant_recipe.py down8 > recipe.json
```

`inspect_quantization.py` is the tool that found two silent failures: a recipe that
matched nothing, and a policy that made the artifact *larger* than the build it was
supposed to beat. Run it on anything you quantize.

Used by the [skills](../skills/) and by the per-model recipes under [models/](../models/).

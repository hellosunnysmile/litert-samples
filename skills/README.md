# Agent Skills

Custom skills, interactive demos, and automation workflow extensions for AI agents.

## On-device model conversion and optimization

Each skill is what was learned by doing the work: the commands that work, the failure
signatures worth recognizing, and the measurements that decided each call. Written for
an agent or a developer picking this up cold. Where an `etf` command appears, the
underlying upstream call is shown too, so nothing here depends on that tool.

| Skill | Answers |
| :-- | :-- |
| [converting-hf-llm-to-litertlm](converting-hf-llm-to-litertlm/SKILL.md) | How do I turn a HuggingFace LLM into a servable `.litertlm`, and why won't my environment build? |
| [choosing-quantization](choosing-quantization/SKILL.md) | Which quantization should I ship, and did my recipe actually do what it says? |
| [making-a-model-run-on-gpu](making-a-model-run-on-gpu/SKILL.md) | Why won't this load on the GPU, and which artifact belongs on which processor? |
| [measuring-model-changes](measuring-model-changes/SKILL.md) | Did that change help, or did I fool myself? |
| [reauthoring-a-model](reauthoring-a-model/SKILL.md) | The tracer can't follow my model — now what, and how do I know the rebuild is faithful? |

Everything quantitative was measured on one Apple Silicon laptop with greedy decoding,
against Qwen3-0.6B and SmolLM2-135M-Instruct. The headline: a mixed-precision build that
keeps the down-projections at 8 bits is **38% smaller, 6% faster and no less accurate**
than a uniform int8 build of the same model — while uniform int4, the obvious way to get
that size, answers 2 of 11 questions instead of 9. Re-run the comparison for your model
and runtime version; none of these numbers are properties of a model alone.

Tools the skills use live in [`utilities/`](../utilities/); the recipes and measured
results per model live under [`models/`](../models/).

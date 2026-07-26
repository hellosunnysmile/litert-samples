# apple-ai-bench — MLX and Core AI, measured the same way

Two Apple-native paths to an on-device model, printing the same JSON contract as
[`litertlm-bench`](../litertlm-bench-swift/) so the three runtimes land in one table:

- **MLX** — our own weights, our own quantization, running on the GPU through Metal.
  This is the like-for-like comparison against LiteRT-LM.
- **Core AI** — Apple's on-device model behind the **FoundationModels** framework
  (`SystemLanguageModel` / `LanguageModelSession`), *not* the older Core ML conversion
  route. There is nothing to convert or quantize: the point of measuring it is that it's
  the baseline every recent iPhone already has.

```bash
swift run apple-ai-bench --runtime mlx --model mlx-community/Qwen3-0.6B-4bit
swift run apple-ai-bench --runtime coreai
```

## What runs where, measured

| Target | MLX | Core AI |
| :-- | :-- | :-- |
| This Mac (M1 Pro, macOS 26.5) | **200.8 tok/s**, 1.17 s TTFT | **unavailable — `deviceNotEligible`** |
| iOS Simulator | no GPU for MLX | expected unavailable |
| iPhone 17 Pro | ready, needs signing | ready, needs signing |

The MLX number is `mlx-community/Qwen3-0.6B-4bit`, and it cross-checks: ETF's Python
`mlx-lm` benchmark of the same weights on the same machine reported 198–216 tok/s. Two
independent paths agreeing is the only reason to believe either.

**Build Release, or the number is fiction.** The same binary built Debug reports 60.2
tok/s — 3.3x slower — because SwiftPM's default configuration turns off optimization.
Every measurement here uses `-configuration Release`.

`--runtime coreai` reports *why* rather than just failing, because the reasons need
different fixes: `deviceNotEligible` is hardware, `appleIntelligenceNotEnabled` is a
Settings toggle, and `modelNotReady` is a download still in progress.

## Four things that gate this, none of them code

1. **The Metal toolchain is a separate download** in Xcode 26 — without it MLX's shaders
   don't compile and the binary dies at runtime with *"Failed to load the default
   metallib"*. Fix: `xcodebuild -downloadComponent MetalToolchain`.
2. **`swift build` alone won't do it.** MLX's Metal shaders are compiled by an Xcode
   build phase, so the package has to be built with `xcodebuild`, and non-interactively
   that needs `-skipPackagePluginValidation -skipMacroValidation` (Xcode otherwise waits
   for a GUI approval of mlx-swift's CudaBuild plugin and mlx-swift-lm's macros).
3. **The LLM libraries moved.** `MLXLLM` / `MLXLMCommon` are no longer in
   `mlx-swift-examples` — they live in `ml-explore/mlx-swift-lm`, and the
   `#huggingFaceLoadModelContainer` macro expands to code needing `HuggingFace` and
   `Tokenizers`, which the app must depend on itself (`swift-huggingface` and
   `swift-transformers`).
4. **Running on the phone needs a signing identity.** `security find-identity -v -p
   codesigning` reporting *0 valid identities* means adding an Apple ID in Xcode →
   Settings → Accounts first. Nothing else blocks the device path — the iPhone is
   detected, trusted, and Apple-Intelligence capable.

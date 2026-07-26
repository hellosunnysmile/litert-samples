// swift-tools-version: 6.0
import PackageDescription

// Benchmarks for the two Apple-native paths to an on-device model, next to LiteRT-LM's:
//
//   * MLX      — Apple's array framework; runs our weights on the GPU via Metal.
//   * Core AI  — Apple's on-device model behind the FoundationModels framework
//                (`SystemLanguageModel` / `LanguageModelSession`), not the older
//                Core ML conversion route.
//
// Both print the same JSON contract as litertlm-bench, so the three can be compared in
// one table instead of three write-ups.
let package = Package(
  name: "apple-ai-bench",
  platforms: [.iOS(.v18), .macOS(.v15)],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
    // The #huggingFaceLoadModelContainer macro expands to code that needs these types;
    // mlx-swift-lm deliberately leaves the Hub client to the app.
    .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.1.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0")
  ],
  targets: [
    .executableTarget(
      name: "apple-ai-bench",
      dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    )
  ]
)

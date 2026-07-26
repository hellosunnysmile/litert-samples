// Benchmark MLX and Apple's on-device model (Core AI / FoundationModels), printing the
// same JSON shape as litertlm-bench so all three runtimes land in one comparison.

import Foundation

#if canImport(FoundationModels)
  import FoundationModels
#endif
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

struct Options {
  var runtime = "mlx"           // mlx | coreai
  var model = "mlx-community/Qwen3-0.6B-4bit"
  var prompt = "Write three sentences about why on-device AI matters."
  var maxTokens = 64
}

func parse() -> Options {
  var options = Options()
  var arguments = Array(CommandLine.arguments.dropFirst())
  while let flag = arguments.first {
    arguments.removeFirst()
    func value() -> String { arguments.isEmpty ? "" : arguments.removeFirst() }
    switch flag {
    case "--runtime": options.runtime = value()
    case "--model": options.model = value()
    case "--prompt": options.prompt = value()
    case "--max-tokens": options.maxTokens = Int(value()) ?? options.maxTokens
    default: break
    }
  }
  return options
}

func emit(_ metrics: [String: Any]) {
  let data = try! JSONSerialization.data(withJSONObject: metrics, options: [.sortedKeys])
  print(String(data: data, encoding: .utf8)!)
}

/// Apple's own on-device model. There is nothing to load or quantize — the point of
/// measuring it is that it is the baseline every iPhone already has.
func benchmarkCoreAI(_ options: Options) async {
  #if canImport(FoundationModels)
    guard #available(macOS 26.0, iOS 26.0, *) else {
      emit(["runtime": "coreai", "available": false, "reason": "OS too old"])
      return
    }
    let model = SystemLanguageModel.default
    guard model.isAvailable else {
      // Say *why*: 'deviceNotEligible' and 'appleIntelligenceNotEnabled' are different
      // problems, and only one of them is fixable in Settings.
      emit(["runtime": "coreai", "available": false,
            "reason": String(describing: model.availability)])
      return
    }
    do {
      let session = LanguageModelSession(model: model)
      let started = Date()
      var firstToken: Date?
      var tokens = 0
      let stream = session.streamResponse(to: options.prompt)
      for try await _ in stream {
        if firstToken == nil { firstToken = Date() }
        tokens += 1
      }
      let finished = Date()
      let ttft = (firstToken ?? finished).timeIntervalSince(started)
      let decodeSeconds = finished.timeIntervalSince(firstToken ?? started)
      emit([
        "runtime": "coreai", "available": true,
        "ttft_ms": ttft * 1000,
        "decode_tps": decodeSeconds > 0 ? Double(max(tokens - 1, 0)) / decodeSeconds : 0,
        "chunks": tokens,
      ])
    } catch {
      emit(["runtime": "coreai", "available": true, "error": "\(error)"])
    }
  #else
    emit(["runtime": "coreai", "available": false, "reason": "FoundationModels missing"])
  #endif
}

/// MLX running our own weights. Unlike Core AI this is a model we chose and quantized,
/// so it is the like-for-like comparison against LiteRT-LM.
func benchmarkMLX(_ options: Options) async {
  do {
    let loadStarted = Date()
    let container = try await #huggingFaceLoadModelContainer(
      configuration: ModelConfiguration(id: options.model))
    let loadSeconds = Date().timeIntervalSince(loadStarted)

    let session = ChatSession(container)
    let started = Date()
    var firstToken: Date?
    var tokens = 0
    for try await _ in session.streamResponse(to: options.prompt) {
      if firstToken == nil { firstToken = Date() }
      tokens += 1
    }
    let finished = Date()
    let ttft = (firstToken ?? finished).timeIntervalSince(started)
    let decodeSeconds = finished.timeIntervalSince(firstToken ?? started)
    emit([
      "runtime": "mlx", "model": options.model,
      "load_ms": loadSeconds * 1000,
      "ttft_ms": ttft * 1000,
      "decode_tps": decodeSeconds > 0 ? Double(max(tokens - 1, 0)) / decodeSeconds : 0,
      "chunks": tokens,
    ])
  } catch {
    emit(["runtime": "mlx", "error": "\(error)"])
  }
}

let options = parse()
switch options.runtime {
case "coreai": await benchmarkCoreAI(options)
default: await benchmarkMLX(options)
}

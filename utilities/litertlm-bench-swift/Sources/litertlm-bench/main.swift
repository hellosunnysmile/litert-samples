// Benchmark a .litertlm through LiteRT-LM's Swift API and print JSON.
//
// The point of printing JSON is that the harness on the other side (`etf bench`) should
// record a phone's numbers the same way it records the laptop's — same metric names,
// same database — so they can be compared instead of retyped.

import Foundation
import LiteRTLM

struct Options {
  var modelPath = ""
  var backend = "cpu"
  var prefillTokens = 128
  var decodeTokens = 64
  var prompt = "Write three sentences about why on-device AI matters."
}

func parse() -> Options {
  var options = Options()
  var arguments = Array(CommandLine.arguments.dropFirst())
  while let flag = arguments.first {
    arguments.removeFirst()
    func value() -> String { arguments.isEmpty ? "" : arguments.removeFirst() }
    switch flag {
    case "--model": options.modelPath = value()
    case "--backend": options.backend = value()
    case "--prefill": options.prefillTokens = Int(value()) ?? options.prefillTokens
    case "--decode": options.decodeTokens = Int(value()) ?? options.decodeTokens
    case "--prompt": options.prompt = value()
    default: break
    }
  }
  return options
}

let options = parse()
guard !options.modelPath.isEmpty else {
  FileHandle.standardError.write(
    "usage: litertlm-bench --model <path.litertlm> [--backend cpu|gpu] [--prefill N] [--decode N]\n"
      .data(using: .utf8)!)
  exit(2)
}
guard let backend = Backend(rawValue: options.backend) else {
  FileHandle.standardError.write("unknown backend '\(options.backend)'\n".data(using: .utf8)!)
  exit(2)
}

do {
  let info = try await benchmark(
    modelPath: options.modelPath,
    backend: backend,
    prefillTokens: options.prefillTokens,
    decodeTokens: options.decodeTokens,
    prompt: options.prompt
  )
  // Named to match what ETF's perf database already stores, so a phone run and a
  // laptop run land in the same columns.
  let metrics: [String: Any] = [
    "backend": options.backend,
    "load_ms": info.initTimeInSecond * 1000,
    "ttft_ms": info.timeToFirstTokenInSecond * 1000,
    "decode_tps": info.lastDecodeTokensPerSecond,
    "prefill_tps": info.lastPrefillTokensPerSecond,
    "prefill_tokens": info.lastPrefillTokenCount,
    "decode_tokens": info.lastDecodeTokenCount,
  ]
  let data = try JSONSerialization.data(withJSONObject: metrics, options: [.sortedKeys])
  print(String(data: data, encoding: .utf8)!)
} catch {
  FileHandle.standardError.write("benchmark failed: \(error)\n".data(using: .utf8)!)
  exit(1)
}

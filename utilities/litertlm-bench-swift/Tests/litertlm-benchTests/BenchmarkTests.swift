// The same measurement as the command-line runner, in a form Xcode can run on a
// simulator or a device.
//
// A benchmark is the one thing you cannot fake on a phone: the whole point is what the
// hardware does. So the code that runs there is the code that runs here, and the model
// path comes in from outside rather than being baked in.

import XCTest

@testable import LiteRTLM

final class BenchmarkTests: XCTestCase {

  /// Set `TEST_RUNNER_LITERTLM_MODEL` to a `.litertlm` path readable by the target.
  /// On the simulator that can be a host path; on a device it has to be inside the app.
  /// xcodebuild forwards `TEST_RUNNER_FOO` as `FOO` for app-hosted tests but passes it
  /// through unchanged to a SwiftPM test bundle, so accept either spelling.
  private func setting(_ name: String) -> String? {
    let environment = ProcessInfo.processInfo.environment
    return environment[name] ?? environment["TEST_RUNNER_\(name)"]
  }

  func testBenchmarkModel() async throws {
    guard let modelPath = setting("LITERTLM_MODEL") else {
      throw XCTSkip("set TEST_RUNNER_LITERTLM_MODEL to a .litertlm path")
    }
    XCTAssertTrue(FileManager.default.fileExists(atPath: modelPath),
                  "model not readable from this target: \(modelPath)")

    let backendName = setting("LITERTLM_BACKEND") ?? "cpu"
    let backend = try XCTUnwrap(Backend(rawValue: backendName))

    let info = try await benchmark(
      modelPath: modelPath,
      backend: backend,
      prefillTokens: Int(setting("LITERTLM_PREFILL") ?? "") ?? 128,
      decodeTokens: Int(setting("LITERTLM_DECODE") ?? "") ?? 64,
      prompt: "Write three sentences about why on-device AI matters."
    )

    // Printed as one line of JSON with the metric names ETF's perf database uses, so a
    // phone run can be parsed and recorded next to a laptop run.
    let metrics: [String: Any] = [
      "backend": backendName,
      "load_ms": info.initTimeInSecond * 1000,
      "ttft_ms": info.timeToFirstTokenInSecond * 1000,
      "decode_tps": info.lastDecodeTokensPerSecond,
      "prefill_tps": info.lastPrefillTokensPerSecond,
      "prefill_tokens": info.lastPrefillTokenCount,
      "decode_tokens": info.lastDecodeTokenCount,
    ]
    let data = try JSONSerialization.data(withJSONObject: metrics, options: [.sortedKeys])
    print("LITERTLM_BENCH \(String(data: data, encoding: .utf8)!)")

    XCTAssertGreaterThan(info.lastDecodeTokensPerSecond, 0)
  }
}

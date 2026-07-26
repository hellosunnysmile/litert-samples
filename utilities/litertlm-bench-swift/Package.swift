// swift-tools-version: 5.9
import PackageDescription

// A LiteRT-LM benchmark you can script, on macOS and on the iOS Simulator.
//
// LiteRT-LM ships a Swift package whose binary target is the released xcframework, so
// nothing here has to be built from source — `swift run` on a Mac, `xcodebuild test`
// against a simulator or a device, same measurement code either way.
let package = Package(
  name: "litertlm-bench",
  platforms: [.iOS(.v15), .macOS(.v14)],
  dependencies: [
    // Local checkout on purpose: resolving the public URL fails today because a
    // git-lfs object under prebuilt/android_arm64/ is missing from the remote, and
    // SwiftPM clones the whole repo before it ever looks at the binary target.
    //   error: external filter 'git-lfs filter-process' failed
    // Point this at your LiteRT-LM checkout, or at the released xcframework directly.
    .package(path: "/tmp/LiteRT-LM-v0.14.0")
  ],
  targets: [
    .executableTarget(
      name: "litertlm-bench",
      dependencies: [.product(name: "LiteRTLM", package: "LiteRT-LM-v0.14.0")]
    ),
    // Runs the same benchmark where a command line can't reach: simulators and phones.
    //   xcodebuild test -scheme litertlm-bench \
    //     -destination 'platform=iOS Simulator,name=iPhone 16' \
    //     TEST_RUNNER_LITERTLM_MODEL=/path/to/model.litertlm
    .testTarget(
      name: "litertlm-benchTests",
      dependencies: [.product(name: "LiteRTLM", package: "LiteRT-LM-v0.14.0")]
    )
  ]
)

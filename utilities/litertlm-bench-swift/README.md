# litertlm-bench — LiteRT-LM benchmarks you can script

One measurement path for a Mac, an iOS Simulator and an iPhone: LiteRT-LM's Swift
package wraps the released xcframework, which carries `ios-arm64` and
`ios-arm64-simulator` slices plus a macOS build, so the same benchmark code runs
everywhere and prints the same JSON.

```bash
swift run litertlm-bench --model ~/.etf/produced/Qwen3-0.6B-produced_mixed_int4_down8.litertlm
{"backend":"cpu","decode_tokens":32,"decode_tps":41.10,"load_ms":684.13,
 "prefill_tokens":128,"prefill_tps":186.78,"ttft_ms":709.63}
```

The metric names match what ETF's perf database stores, so a phone run and a laptop run
land in the same columns instead of being retyped.

## On a simulator or a phone

A command line can't reach either, so the same benchmark also exists as a test:

```bash
xcodebuild test -scheme litertlm-bench \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  TEST_RUNNER_LITERTLM_MODEL=/path/to/model.litertlm
```

`TEST_RUNNER_LITERTLM_BACKEND` (`cpu`/`gpu`), `..._PREFILL` and `..._DECODE` tune the
run. On the simulator the model can be a host path; on a device it has to be inside the
app bundle or its container.

Two things gate the phone paths, and neither is a code problem:

- **Simulator**: Xcode 26 downloads runtimes separately — `xcodebuild -downloadPlatform iOS`.
- **Device**: `xcodebuild test -destination 'platform=iOS,id=<udid>'` needs a signing
  identity. `security find-identity -v -p codesigning` reporting *0 valid identities*
  means adding an Apple ID to Xcode first.

## Three upstream traps, and what they cost

Worth knowing before you spend an afternoon on them:

1. **`.package(url: "…/LiteRT-LM")` doesn't resolve.** SwiftPM clones the whole repo
   before it looks at the binary target, and a git-lfs object under
   `prebuilt/android_arm64/` is missing from the remote:
   `error: external filter 'git-lfs filter-process' failed`. Use a local checkout, or
   clone with `GIT_LFS_SKIP_SMUDGE=1` — the Swift package needs none of those `.so`s.
2. **The v0.14.0 tag's `Package.swift` has stale checksums** for its own release zips;
   `main` has the correct ones. Symptom:
   `checksum of downloaded artifact … does not match checksum specified by the manifest`.
   Fix by copying the checksums from `main` into the tagged manifest.
3. **`ld: duplicate symbols` in `libswiftCompatibilityPacks.a`** when the package
   declares an older deployment target. Raising it to `.macOS(.v14)` drops the
   compatibility pack and links cleanly.

Also: `main` didn't compile at the time of writing (`ToolManager.swift`:
`cannot infer contextual base in reference to member 'utf8'`), so this pins the v0.14.0
source.

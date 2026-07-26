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
SIM=$(xcrun simctl list devices available -j | ...)      # any booted iPhone
xcrun simctl boot "$SIM"

# xcodebuild's TEST_RUNNER_* forwarding does not reach a SwiftPM test bundle — the
# xctest process inherits the *simulator's* environment, so set it there.
xcrun simctl spawn "$SIM" launchctl setenv LITERTLM_MODEL /path/to/model.litertlm
xcodebuild test -scheme litertlm-bench -destination "platform=iOS Simulator,id=$SIM"
```

```
LITERTLM_BENCH {"backend":"cpu","decode_tps":25.86,"load_ms":1133.78,
                "prefill_tps":70.36,"ttft_ms":1857.99}
```

`LITERTLM_BACKEND` (`cpu`/`gpu`), `LITERTLM_PREFILL` and `LITERTLM_DECODE` tune the run,
set the same way. On the simulator the model can be a host path; on a device it has to be
inside the app bundle or its container.

**What the simulator number is worth:** 25.9 tok/s against 41.1 for the same artifact
run natively on the same Mac. The simulator is a Mac wearing a costume — it is the right
place to prove the code path works and the wrong place to decide what to ship.

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

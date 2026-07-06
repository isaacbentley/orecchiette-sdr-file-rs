# 📁 orecchiette-sdr-file-rs: File-backed SDR Sources

[![CI](https://github.com/isaacbentley/orecchiette-sdr-file-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/orecchiette-sdr-file-rs/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-ab6000.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

## 🎯 **Overview**

File-backed implementations of the
[`SdrSource`](https://github.com/isaacbentley/orecchiette-sdr-source-rs) trait. Two source types
share the crate:

| Source | Inputs | Centre frequency comes from |
|---|---|---|
| [`RawIqFileSource`](#rawiqfilesource) | header-less `.bin` (interleaved `i16`, scaled by `1/32768`) or `.cf32`/`.f32`/anything-else (interleaved IEEE-754 `f32`) | caller (`center_frequency_hz` field — files have no metadata) |
| [`SigmfFileSource`](#sigmffilesource) | [SigMF](https://github.com/sigmf/sigmf-spec) recordings: `.sigmf-meta` (JSON sidecar) + `.sigmf-data` (raw payload) | metadata (`captures[].core:frequency`) |

Both backends are non-hopping — the file *is* the capture, played
back at its natural rate. They accept `DwellAdvice` at the trait
boundary and ignore it.

Each backend streams the data in 1 MB I/O chunks, decodes into
`Complex32`, and emits 262 144-sample `IqPacket`s to match the
SDR applications worker pool's batch expectation. Partial samples at chunk
boundaries are stitched across reads so no IQ pair is split.

## 📄 **RawIqFileSource**

```rust,ignore
use orecchiette_sdr_file_rs::RawIqFileSource;
use orecchiette_sdr_source_rs::{DwellAdvice, SdrSource, SourceConfig};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// File backends ignore DwellAdvice but the trait still requires one.
struct NoSignalLog;
impl DwellAdvice for NoSignalLog {
    fn latest_signal_at(&self, _freq_key_khz: u64) -> Option<Instant> { None }
}
let advice: Arc<dyn DwellAdvice> = Arc::new(NoSignalLog);

let source = Box::new(RawIqFileSource {
    paths: vec![PathBuf::from("captures/drone_sample_2G.bin")],
    center_frequency_hz: 2.435e9,
});
let handle = source.start(config, advice).unwrap();
```

The format is decided by the file's extension:

- `.bin` → interleaved `i16` little-endian, scaled by `1 / 32768.0` on
  decode (matches the Tampere Zenodo dataset and most "raw ADC dump"
  recordings).
- anything else → interleaved IEEE-754 `f32` little-endian, taken as-is.

If you pass multiple paths, they're played sequentially.
`config.sample_rate_hz` is the assumed playback rate; the caller is
responsible for setting it to match how the file was recorded.

## 📑 **SigmfFileSource**

```rust,ignore
use orecchiette_sdr_file_rs::SigmfFileSource;
use orecchiette_sdr_source_rs::{SdrSource, SourceConfig};
use std::path::PathBuf;
// `advice` constructed as in the RawIqFileSource example above.

let source = Box::new(SigmfFileSource {
    paths: vec![
        PathBuf::from("captures/drone.sigmf-meta"),   // either side OK
    ],
});
let handle = source.start(config, advice).unwrap();
```

For each path, the source:

1. Resolves the `(meta, data)` pair via `sigmf::resolve_pair()` — the
   caller can hand it the `.sigmf-meta`, the `.sigmf-data`, or the
   bare basename and the resolver figures out the sibling.
2. Loads + validates the metadata (`SigmfMetadata::load()`).
3. Picks the datatype (`global.core:datatype`):
   - `cf32_le` → interleaved IEEE-754 f32, 8 bytes per IQ pair.
   - `ci16_le` → interleaved i16 LE, 4 bytes per IQ pair, scaled by `1/32768`.
   - anything else fails fast with a typed error.
4. Streams `.sigmf-data` and emits packets tagged with
   `captures[].core:frequency` (uses the first capture; multi-capture
   recordings are read end-to-end). Sample rate comes from
   `global.core:sample_rate`.

The caller's `config.sample_rate_hz` and any per-packet centre-frequency
guess are ignored — SigMF metadata is the source of truth.

## 🧪 **Tests**

```bash
cargo test -p orecchiette-sdr-file-rs
```

14 tests cover:

- raw `i16`/`f32` decoders (round-trip + scaling).
- SigMF metadata parse: minimal, full, unknown namespaces, missing
  captures, unsupported datatype rejection.
- `resolve_pair` from either side and the bare basename; the missing-
  sibling error path.
- `looks_like_sigmf` recognises `.sigmf-meta` / `.sigmf-data` / bare
  base names and rejects unrelated extensions.
- End-to-end `sigmf_file_source_round_trip` writes a synthetic
  `cf32_le` SigMF pair, runs it through `SigmfFileSource`, and drains
  three packets, verifying centre frequency and sample rate
  propagate from metadata.

## 📦 **Dependencies**

```toml
orecchiette-sdr-source-rs = { git = "https://github.com/isaacbentley/orecchiette-sdr-source-rs.git", branch = "main" }
crossbeam     = "0.8"
num-complex   = "0.4"
anyhow        = "1.0"
tracing       = "0.1"
serde         = { version = "1.0", features = ["derive"] }
serde_json    = "1.0"

[dev-dependencies]
tempfile      = "3"
```

## 📚 **Documentation**

- [Architecture & Design](DESIGN.md) — internal architecture and execution flow.

## 📄 **License**

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) - see the [LICENSE](../../LICENSE) file for details.

## 📞 **Support**

- 🐛 **Issues**: [GitHub Issues](https://github.com/isaacbentley/orecchiette-sdr-file-rs/issues)

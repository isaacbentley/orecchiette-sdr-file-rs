# Contributing to orecchiette-sdr-file-rs

First off, thank you for considering contributing! This crate
implements the `SdrSource` trait (from `orecchiette-sdr-source-rs`)
for two file-backed sources: raw interleaved IQ and SigMF recordings.

## Quick Start

```bash
git clone https://github.com/isaacbentley/orecchiette-sdr-file-rs.git
cd orecchiette-sdr-file-rs

cargo test
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
cargo deny check
```

## Testing Your Changes

Tests use synthetic fixtures generated on the fly (no large capture
files checked in). If you're touching the SigMF metadata parser or
the raw-IQ decoders, please add a round-trip test alongside your
change rather than only testing manually.

## Code Style

We use standard `rustfmt` defaults. Please run `cargo fmt --all` before pushing.

Clippy is run with `-D warnings` in CI. If a lint is genuinely wrong for the situation, allow it with a `// ALLOW:` justification comment explaining why.

## Pull Requests

- **Commit messages:** Describe *why* the change is needed and *what* it changes.
- **Templates:** Please fill out the Pull Request template when opening a PR.

## License

By contributing, you agree your contributions will be licensed under GPL-3.0-or-later, the same as the rest of the project.

//! SigMF (Signal Metadata Format) support.
//!
//! Implements just enough of the [SigMF core spec][spec] to load
//! captures produced by common SDR tooling (gr-sigmf, GQRX, SDR++,
//! Aaronia RTSA-Suite export, etc.):
//!
//! * Parses the JSON `.sigmf-meta` file into [`SigmfMetadata`].
//! * Decodes the paired `.sigmf-data` payload as one of the
//!   [`DataType`] variants.
//! * Surfaces the global sample rate and the first capture's centre
//!   frequency as the canonical values for downstream IQ-packet
//!   tagging.
//!
//! Out of scope for this version:
//!
//! * `.sigmf` tar archives — only the unpacked meta+data file pair
//!   is supported.
//! * `.sigmf-collection` multi-recording metadata.
//! * Annotations (per-sample marks).
//! * Per-capture frequency switching mid-stream — the first
//!   capture's frequency tags every emitted packet.
//! * Datatypes other than `cf32_le`, `ci16_le`, and `ci8` (the formats
//!   that account for nearly every real-world capture). The reader
//!   surfaces a clear "unsupported datatype" error for the rest.
//!
//! [spec]: https://github.com/sigmf/sigmf-spec

use anyhow::{Context, Result, anyhow};
use num_complex::Complex32;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Top-level SigMF metadata, mirroring the JSON layout defined by
/// the spec at <https://github.com/sigmf/sigmf-spec>.
///
/// Extension namespaces (anything prefixed other than `core:`) are
/// ignored — `serde_json` skips unknown fields by default.
#[derive(Debug, Clone, Deserialize)]
pub struct SigmfMetadata {
    pub global: Global,
    #[serde(default)]
    pub captures: Vec<Capture>,
    /// Annotations are present in the spec but not consumed here.
    /// Kept as a count so callers can warn if the file has annotation
    /// data they expected us to respect.
    #[serde(default, skip_serializing)]
    pub annotations: Vec<serde_json::Value>,
}

/// Global object — applies to the entire recording.
#[derive(Debug, Clone, Deserialize)]
pub struct Global {
    #[serde(rename = "core:datatype")]
    pub datatype: String,
    #[serde(rename = "core:sample_rate")]
    pub sample_rate: f64,
    #[serde(rename = "core:version")]
    pub version: String,
    #[serde(rename = "core:description", default)]
    pub description: Option<String>,
    #[serde(rename = "core:hw", default)]
    pub hardware: Option<String>,
    #[serde(rename = "core:author", default)]
    pub author: Option<String>,
}

/// One contiguous span of samples in the data file. Recordings with
/// frequency-hopping or gain-stepping use multiple captures, each
/// tagged with their own `sample_start` index.
#[derive(Debug, Clone, Deserialize)]
pub struct Capture {
    #[serde(rename = "core:sample_start", default)]
    pub sample_start: u64,
    #[serde(rename = "core:frequency", default)]
    pub frequency: Option<f64>,
    #[serde(rename = "core:datetime", default)]
    pub datetime: Option<String>,
}

/// Supported SigMF datatypes. SigMF allows many more (signed/unsigned
/// integer widths from 8 to 64 bits, big-endian variants, real-only,
/// etc.) — this enum is the subset we know how to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// `cf32_le` — interleaved IEEE-754 32-bit floats, little-endian.
    /// Most common for SDR captures produced by gr-sigmf and GQRX.
    Cf32Le,
    /// `ci16_le` — interleaved signed 16-bit integers, little-endian.
    /// Scaled by `1 / 32768.0` on decode so the resulting `Complex32`
    /// matches the unit-disc convention every other source emits.
    Ci16Le,
    /// `ci8` — interleaved signed 8-bit integers.
    Ci8,
}

impl DataType {
    /// Number of bytes per IQ pair on disk.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            DataType::Cf32Le => 8, // 2 × f32
            DataType::Ci16Le => 4, // 2 × i16
            DataType::Ci8 => 2,    // 2 × i8
        }
    }

    /// Decode a complete byte buffer into `Complex32` samples.
    /// The buffer length must be a multiple of `bytes_per_sample()`;
    /// trailing partial bytes are silently dropped.
    pub fn decode(self, bytes: &[u8]) -> Vec<Complex32> {
        let n = self.bytes_per_sample();
        let pairs = bytes.len() / n;
        let mut out = Vec::with_capacity(pairs);
        for chunk in bytes.chunks_exact(n) {
            out.push(match self {
                DataType::Cf32Le => {
                    let re = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let im = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    Complex32::new(re, im)
                }
                DataType::Ci16Le => {
                    let re = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                    let im = i16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 32768.0;
                    Complex32::new(re, im)
                }
                DataType::Ci8 => {
                    let re = (chunk[0] as i8) as f32 / 127.0;
                    let im = (chunk[1] as i8) as f32 / 127.0;
                    Complex32::new(re, im)
                }
            });
        }
        out
    }

    /// Parse a SigMF `core:datatype` string into a [`DataType`].
    /// Returns `Err` for any spec-valid string we don't implement.
    pub fn from_spec(s: &str) -> Result<Self> {
        match s {
            "cf32_le" | "cf32" => Ok(DataType::Cf32Le),
            "ci16_le" | "ci16" => Ok(DataType::Ci16Le),
            "ci8" | "ci8_le" => Ok(DataType::Ci8),
            other => Err(anyhow!(
                "unsupported SigMF datatype {other:?} (this reader handles cf32_le, ci16_le, and ci8)"
            )),
        }
    }
}

impl SigmfMetadata {
    /// Load a `.sigmf-meta` JSON file from disk.
    pub fn load(meta_path: &Path) -> Result<Self> {
        let body = fs::read_to_string(meta_path)
            .with_context(|| format!("read SigMF meta {}", meta_path.display()))?;
        serde_json::from_str(&body)
            .with_context(|| format!("parse SigMF meta {}", meta_path.display()))
    }

    /// First-capture centre frequency in Hz, or `None` if no captures
    /// declare one. The orchestrator falls back to the user's
    /// `--center-freq` for raw recordings missing this field.
    pub fn center_frequency_hz(&self) -> Option<f64> {
        self.captures.first().and_then(|c| c.frequency)
    }

    /// Global sample rate in Hz. Mandatory per the spec.
    pub fn sample_rate_hz(&self) -> f64 {
        self.global.sample_rate
    }

    /// Resolved [`DataType`] or a spec-derived error.
    pub fn data_type(&self) -> Result<DataType> {
        DataType::from_spec(&self.global.datatype)
    }
}

/// Given a SigMF-related path (`.sigmf-meta`, `.sigmf-data`, or the
/// bare basename with no extension), return the `(meta, data)` pair.
/// The orchestrator passes whichever filename the user globbed.
pub fn resolve_pair(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let s = path.to_string_lossy();
    let (meta, data) = if let Some(base) = s.strip_suffix(".sigmf-meta") {
        (
            path.to_path_buf(),
            PathBuf::from(format!("{base}.sigmf-data")),
        )
    } else if let Some(base) = s.strip_suffix(".sigmf-data") {
        (
            PathBuf::from(format!("{base}.sigmf-meta")),
            path.to_path_buf(),
        )
    } else {
        // Bare basename: assume both siblings exist.
        (
            PathBuf::from(format!("{s}.sigmf-meta")),
            PathBuf::from(format!("{s}.sigmf-data")),
        )
    };
    if !meta.exists() {
        return Err(anyhow!(
            "SigMF metadata file not found: {} (looked up from {})",
            meta.display(),
            path.display()
        ));
    }
    if !data.exists() {
        return Err(anyhow!(
            "SigMF data file not found: {} (looked up from {})",
            data.display(),
            path.display()
        ));
    }
    Ok((meta, data))
}

/// `true` if the path looks like a SigMF artefact (meta, data, or
/// a recording basename with both siblings present).
pub fn looks_like_sigmf(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.ends_with(".sigmf-meta") || s.ends_with(".sigmf-data") {
        return true;
    }
    // Bare basename with both siblings present.
    PathBuf::from(format!("{s}.sigmf-meta")).exists()
        && PathBuf::from(format!("{s}.sigmf-data")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_meta(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(format!("{name}.sigmf-meta"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn datatype_from_spec_recognises_supported() {
        assert_eq!(DataType::from_spec("cf32_le").unwrap(), DataType::Cf32Le);
        assert_eq!(DataType::from_spec("cf32").unwrap(), DataType::Cf32Le);
        assert_eq!(DataType::from_spec("ci16_le").unwrap(), DataType::Ci16Le);
        assert_eq!(DataType::from_spec("ci16").unwrap(), DataType::Ci16Le);
        assert_eq!(DataType::from_spec("ci8_le").unwrap(), DataType::Ci8);
        assert_eq!(DataType::from_spec("ci8").unwrap(), DataType::Ci8);
    }

    #[test]
    fn datatype_from_spec_rejects_unsupported() {
        // The full SigMF spec includes many we haven't implemented:
        for unsupported in &["cu8", "cf64_le", "ri16", "ci32_be", "cu16_le", "garbage"] {
            assert!(
                DataType::from_spec(unsupported).is_err(),
                "expected {unsupported:?} to be rejected"
            );
        }
    }

    #[test]
    fn cf32_le_decode_round_trip() {
        // Two IQ pairs as LE f32 bytes.
        let samples = [Complex32::new(1.5, -2.5), Complex32::new(0.25, 0.5)];
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for s in samples {
            bytes.extend_from_slice(&s.re.to_le_bytes());
            bytes.extend_from_slice(&s.im.to_le_bytes());
        }
        let decoded = DataType::Cf32Le.decode(&bytes);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0].re - 1.5).abs() < 1e-6);
        assert!((decoded[0].im + 2.5).abs() < 1e-6);
        assert!((decoded[1].re - 0.25).abs() < 1e-6);
        assert!((decoded[1].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ci16_le_decode_scales_to_unit_disc() {
        // int16 32767 → ~1.0, -32768 → ~-1.0.
        let bytes = [
            0xFF, 0x7F, // re = 32767
            0x00, 0x80, // im = -32768
        ];
        let decoded = DataType::Ci16Le.decode(&bytes);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0].re - (32767.0 / 32768.0)).abs() < 1e-6);
        assert!((decoded[0].im + 1.0).abs() < 1e-6);
    }

    #[test]
    fn ci8_decode_scales_to_unit_disc() {
        let bytes = [
            127u8,        // re = 127
            -127i8 as u8, // im = -127
        ];
        let decoded = DataType::Ci8.decode(&bytes);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0].re - 1.0).abs() < 1e-6);
        assert!((decoded[0].im + 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_minimal_metadata() {
        let body = r#"{
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1000000,
                "core:version": "1.0.0"
            },
            "captures": [
                { "core:sample_start": 0, "core:frequency": 2435000000 }
            ],
            "annotations": []
        }"#;
        let meta: SigmfMetadata = serde_json::from_str(body).unwrap();
        assert_eq!(meta.sample_rate_hz(), 1_000_000.0);
        assert_eq!(meta.center_frequency_hz(), Some(2_435_000_000.0));
        assert_eq!(meta.data_type().unwrap(), DataType::Cf32Le);
        assert_eq!(meta.global.version, "1.0.0");
    }

    #[test]
    fn parse_metadata_with_no_captures() {
        // The spec requires captures, but accept its absence — we
        // simply have no centre frequency to tag packets with.
        let body = r#"{
            "global": {
                "core:datatype": "ci16_le",
                "core:sample_rate": 2400000,
                "core:version": "1.0.0",
                "core:description": "test recording"
            }
        }"#;
        let meta: SigmfMetadata = serde_json::from_str(body).unwrap();
        assert_eq!(meta.sample_rate_hz(), 2_400_000.0);
        assert!(meta.center_frequency_hz().is_none());
        assert_eq!(meta.global.description.as_deref(), Some("test recording"));
    }

    #[test]
    fn parse_metadata_ignores_unknown_namespaces() {
        // Extensions (anything not in the `core:` namespace) must round-trip
        // through serde without exploding.
        let body = r#"{
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1000000,
                "core:version": "1.0.0",
                "antenna:gain": 30,
                "geo:location": [0.0, 0.0]
            },
            "captures": [{ "core:sample_start": 0 }]
        }"#;
        let meta: SigmfMetadata = serde_json::from_str(body).unwrap();
        assert_eq!(meta.sample_rate_hz(), 1_000_000.0);
    }

    #[test]
    fn resolve_pair_from_meta_path() {
        let dir = tempfile::tempdir().unwrap();
        let _ = write_meta(
            dir.path(),
            "capture1",
            r#"{"global":{"core:datatype":"cf32_le","core:sample_rate":1,"core:version":"1.0.0"}}"#,
        );
        let data_path = dir.path().join("capture1.sigmf-data");
        std::fs::write(&data_path, [0u8; 16]).unwrap();

        let (meta, data) = resolve_pair(&dir.path().join("capture1.sigmf-meta")).unwrap();
        assert!(meta.to_string_lossy().ends_with(".sigmf-meta"));
        assert!(data.to_string_lossy().ends_with(".sigmf-data"));
    }

    #[test]
    fn resolve_pair_from_data_path() {
        let dir = tempfile::tempdir().unwrap();
        let _ = write_meta(
            dir.path(),
            "capture2",
            r#"{"global":{"core:datatype":"cf32_le","core:sample_rate":1,"core:version":"1.0.0"}}"#,
        );
        let data_path = dir.path().join("capture2.sigmf-data");
        std::fs::write(&data_path, [0u8; 16]).unwrap();

        let (meta, data) = resolve_pair(&data_path).unwrap();
        assert!(meta.to_string_lossy().ends_with(".sigmf-meta"));
        assert!(data.to_string_lossy().ends_with(".sigmf-data"));
    }

    #[test]
    fn resolve_pair_errors_on_missing_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let meta_path = write_meta(
            dir.path(),
            "lonely",
            r#"{"global":{"core:datatype":"cf32_le","core:sample_rate":1,"core:version":"1.0.0"}}"#,
        );
        // No sibling .sigmf-data — resolve should report it.
        let err = resolve_pair(&meta_path).unwrap_err();
        assert!(
            err.to_string().contains("data file not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn looks_like_sigmf_recognises_paths() {
        let dir = tempfile::tempdir().unwrap();
        let _ = write_meta(
            dir.path(),
            "rec",
            r#"{"global":{"core:datatype":"cf32_le","core:sample_rate":1,"core:version":"1.0.0"}}"#,
        );
        std::fs::write(dir.path().join("rec.sigmf-data"), [0u8; 16]).unwrap();

        assert!(looks_like_sigmf(&dir.path().join("rec.sigmf-meta")));
        assert!(looks_like_sigmf(&dir.path().join("rec.sigmf-data")));
        assert!(looks_like_sigmf(&dir.path().join("rec")));
        assert!(!looks_like_sigmf(&dir.path().join("random.iq")));
    }
}

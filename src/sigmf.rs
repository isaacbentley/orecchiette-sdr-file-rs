//! SigMF (Signal Metadata Format) support.
//!
//! Implements just enough of the [SigMF core spec][spec] to load
//! captures produced by common SDR tooling (gr-sigmf, GQRX, SDR++,
//! Aaronia RTSA-Suite export, etc.):
//!
//! * Parses the JSON `.sigmf-meta` file into [`SigmfMetadata`].
//! * Decodes the paired `.sigmf-data` payload as one of the
//!   [`DataType`] variants.
//! * Surfaces the global sample rate and each capture's centre
//!   frequency for downstream IQ-packet tagging. Playback cuts packets
//!   at every capture's `core:sample_start`, so a multi-capture
//!   recording is labelled capture by capture.
//!
//! Out of scope for this version:
//!
//! * `.sigmf` tar archives — only the unpacked meta+data file pair
//!   is supported.
//! * `.sigmf-collection` multi-recording metadata.
//! * Multichannel datasets (`core:num_channels` other than 1 are rejected).
//! * Annotations (per-sample marks).
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
#[derive(Debug, Clone)]
pub struct Global {
    pub datatype: String,
    pub sample_rate: f64,
    /// Missing versions default to empty for compatibility with older writers.
    pub version: String,
    pub description: Option<String>,
    pub hardware: Option<String>,
    pub author: Option<String>,
}

impl<'de> Deserialize<'de> for Global {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Keep the public fields unchanged while validating the on-disk layout.
        #[derive(Deserialize)]
        struct Fields {
            #[serde(rename = "core:datatype")]
            datatype: String,
            #[serde(rename = "core:sample_rate")]
            sample_rate: f64,
            #[serde(rename = "core:version", default)]
            version: String,
            #[serde(rename = "core:description", default)]
            description: Option<String>,
            #[serde(rename = "core:hw", default)]
            hardware: Option<String>,
            #[serde(rename = "core:author", default)]
            author: Option<String>,
            #[serde(rename = "core:num_channels", default = "single_channel")]
            num_channels: u64,
        }
        fn single_channel() -> u64 {
            1
        }
        let fields = Fields::deserialize(deserializer)?;
        if fields.num_channels != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported core:num_channels {}: only single-channel recordings are supported",
                fields.num_channels
            )));
        }
        Ok(Self {
            datatype: fields.datatype,
            sample_rate: fields.sample_rate,
            version: fields.version,
            description: fields.description,
            hardware: fields.hardware,
            author: fields.author,
        })
    }
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
    /// `ci8` — interleaved signed 8-bit integers, scaled by `1 / 128.0`
    /// — the same full scale as live HackRF input, so a recording
    /// replays at the level it was captured at.
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
        let mut out = Vec::new();
        self.decode_into(bytes, &mut out);
        out
    }

    /// [`DataType::decode`], appending to `out` so a caller can reuse
    /// one buffer across reads.
    pub fn decode_into(self, bytes: &[u8], out: &mut Vec<Complex32>) {
        let chunks = bytes.chunks_exact(self.bytes_per_sample());
        // One loop per datatype, so the per-sample body is branch-free.
        match self {
            DataType::Cf32Le => out.extend(chunks.map(|c| {
                let re = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let im = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                Complex32::new(re, im)
            })),
            DataType::Ci16Le => out.extend(chunks.map(|c| {
                let re = i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
                let im = i16::from_le_bytes([c[2], c[3]]) as f32 / 32768.0;
                Complex32::new(re, im)
            })),
            DataType::Ci8 => out.extend(chunks.map(|c| {
                let re = (c[0] as i8) as f32 / 128.0;
                let im = (c[1] as i8) as f32 / 128.0;
                Complex32::new(re, im)
            })),
        }
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

impl DataType {
    /// Inverse of [`DataType::from_spec`] — the `core:datatype` string
    /// this variant should be tagged with on write.
    pub fn to_spec(self) -> &'static str {
        match self {
            DataType::Cf32Le => "cf32_le",
            DataType::Ci16Le => "ci16_le",
            DataType::Ci8 => "ci8",
        }
    }
}

/// One capture-array entry for [`SigmfWriter::finalize`]. Mirrors
/// [`Capture`] but as an owned builder, since the writer doesn't parse
/// an existing file.
#[derive(Debug, Clone, Default)]
pub struct SigmfCapture {
    pub sample_start: u64,
    pub frequency_hz: Option<f64>,
    pub datetime_rfc3339: Option<String>,
    /// GeoJSON `Point` coordinates as `[longitude, latitude]`.
    pub geolocation: Option<[f64; 2]>,
}

/// Global-object fields for [`SigmfWriter::finalize`].
#[derive(Debug, Clone)]
pub struct SigmfWriterMeta {
    pub sample_rate_hz: f64,
    pub hardware: Option<String>,
    pub description: Option<String>,
    pub recorder: Option<String>,
    pub captures: Vec<SigmfCapture>,
    /// Passed through verbatim as the top-level `annotations` array.
    /// The reader (`SigmfMetadata`) doesn't parse annotation content —
    /// callers that want documented sample ranges (e.g. ground-truth
    /// labels for a synthetic fixture) can still emit them here.
    pub annotations: Vec<serde_json::Value>,
}

/// Shared `.sigmf-data` + `.sigmf-meta` writer. Centralizes the buffer
/// capacity, sample encoding, and metadata JSON schema that every
/// SigMF-producing binary in this ecosystem otherwise hand-rolls (and
/// had already drifted on — differing `BufWriter` sizes, one-off
/// `unsafe` byte casts, and ad hoc metadata field sets).
pub struct SigmfWriter {
    data: std::io::BufWriter<fs::File>,
    meta_path: PathBuf,
    datatype: DataType,
}

/// Matches the `BufWriter` capacity every existing recorder already
/// converged on for real-time capture throughput.
const WRITER_BUFFER_BYTES: usize = 16 * 1024 * 1024;

impl SigmfWriter {
    /// Create a new `.sigmf-data` file (and remember where its sibling
    /// `.sigmf-meta` belongs) for `base_path`, e.g. `base_path =
    /// "capture"` produces `capture.sigmf-data` / `capture.sigmf-meta`.
    /// Dots in basenames are preserved; an existing `.sigmf-data` or
    /// `.sigmf-meta` suffix is normalized to the same recording pair.
    pub fn create(base_path: &Path, datatype: DataType) -> Result<Self> {
        let (meta_path, data_path) = pair_paths(base_path);
        let file = fs::File::create(&data_path)
            .with_context(|| format!("create SigMF data file {}", data_path.display()))?;
        Ok(Self {
            data: std::io::BufWriter::with_capacity(WRITER_BUFFER_BYTES, file),
            meta_path,
            datatype,
        })
    }

    /// Write already-encoded wire bytes straight through — the fast
    /// path for sources (e.g. HackRF's native `ci8`) that hand back
    /// bytes in the on-disk format with nothing to decode.
    pub fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        use std::io::Write;
        self.data.write_all(bytes).context("write SigMF data bytes")
    }

    /// Encode and write `Complex32` samples per this writer's
    /// [`DataType`]. `Cf32Le` uses a byte view on little-endian hosts
    /// and explicit little-endian encoding elsewhere; `Ci16Le`/`Ci8`
    /// scale from the unit disc by the same full scale the decoder
    /// divides by (32768 / 128), rounding to nearest and saturating, so
    /// decoding what was written returns every representable value
    /// exactly. (Truncation with `as` biased every sample toward zero.)
    ///
    /// An integer datatype has no code for NaN or infinity, so a
    /// non-finite sample is an error and nothing from that call is
    /// written: `NaN as i16` is 0, which would record silence that was
    /// never received, and saturating an infinity would record a
    /// full-scale sample that was never received either. `Cf32Le`
    /// represents both and writes them as given.
    pub fn write_samples(&mut self, samples: &[Complex32]) -> Result<()> {
        if self.datatype != DataType::Cf32Le
            && let Some(i) = samples
                .iter()
                .position(|s| !s.re.is_finite() || !s.im.is_finite())
        {
            anyhow::bail!(
                "sample {i} is {} and {} has no code for a non-finite value",
                samples[i],
                self.datatype.to_spec()
            );
        }
        match self.datatype {
            #[cfg(target_endian = "big")]
            DataType::Cf32Le => self.write_cf32_le_portable(samples),
            #[cfg(target_endian = "little")]
            DataType::Cf32Le => {
                // SAFETY: `Complex32` (`num_complex::Complex<f32>`) is
                // `#[repr(C)]` with two `f32` fields and no padding, so
                // reading it as `2 * size_of::<f32>()` bytes per
                // element is a valid reinterpretation. `samples` is a
                // borrowed, already-initialized slice, so the byte
                // view covers only initialized memory.
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        samples.as_ptr() as *const u8,
                        std::mem::size_of_val(samples),
                    )
                };
                self.write_raw(bytes)
            }
            DataType::Ci16Le => {
                let mut buf = Vec::with_capacity(samples.len() * 4);
                for s in samples {
                    let re = (s.re * 32768.0)
                        .round()
                        .clamp(i16::MIN as f32, i16::MAX as f32)
                        as i16;
                    let im = (s.im * 32768.0)
                        .round()
                        .clamp(i16::MIN as f32, i16::MAX as f32)
                        as i16;
                    buf.extend_from_slice(&re.to_le_bytes());
                    buf.extend_from_slice(&im.to_le_bytes());
                }
                self.write_raw(&buf)
            }
            DataType::Ci8 => {
                let mut buf = Vec::with_capacity(samples.len() * 2);
                for s in samples {
                    let re = (s.re * 128.0).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8;
                    let im = (s.im * 128.0).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8;
                    buf.push(re as u8);
                    buf.push(im as u8);
                }
                self.write_raw(&buf)
            }
        }
    }

    // Also compiled in tests so the big-endian host's encoding path is
    // exercised on little-endian CI machines.
    #[cfg(any(target_endian = "big", test))]
    fn write_cf32_le_portable(&mut self, samples: &[Complex32]) -> Result<()> {
        let mut buffer = [0u8; 64 * 1024];
        for chunk in samples.chunks(buffer.len() / 8) {
            for (sample, bytes) in chunk.iter().zip(buffer.as_chunks_mut::<8>().0.iter_mut()) {
                bytes[..4].copy_from_slice(&sample.re.to_le_bytes());
                bytes[4..].copy_from_slice(&sample.im.to_le_bytes());
            }
            self.write_raw(&buffer[..chunk.len() * 8])?;
        }
        Ok(())
    }

    /// Flush the data file and write the paired `.sigmf-meta` JSON.
    /// Rejects sample rates that cannot be played back and non-finite
    /// capture frequencies or geolocation coordinates.
    pub fn finalize(mut self, meta: SigmfWriterMeta) -> Result<()> {
        use std::io::Write;
        crate::packet_sample_rate(meta.sample_rate_hz).map_err(|e| anyhow!(e))?;
        for capture in &meta.captures {
            if capture.frequency_hz.is_some_and(|f| !f.is_finite()) {
                return Err(anyhow!("capture frequency must be finite"));
            }
            if capture
                .geolocation
                .is_some_and(|point| point.iter().any(|v| !v.is_finite()))
            {
                return Err(anyhow!("capture geolocation coordinates must be finite"));
            }
        }
        self.data.flush().context("flush SigMF data file")?;

        let mut global = serde_json::json!({
            "core:datatype": self.datatype.to_spec(),
            "core:sample_rate": meta.sample_rate_hz,
            "core:version": "1.0.0",
        });
        if let Some(hw) = &meta.hardware {
            global["core:hw"] = serde_json::json!(hw);
        }
        if let Some(desc) = &meta.description {
            global["core:description"] = serde_json::json!(desc);
        }
        if let Some(rec) = &meta.recorder {
            global["core:recorder"] = serde_json::json!(rec);
        }

        let captures: Vec<serde_json::Value> = meta
            .captures
            .iter()
            .map(|c| {
                let mut v = serde_json::json!({ "core:sample_start": c.sample_start });
                if let Some(f) = c.frequency_hz {
                    v["core:frequency"] = serde_json::json!(f);
                }
                if let Some(dt) = &c.datetime_rfc3339 {
                    v["core:datetime"] = serde_json::json!(dt);
                }
                if let Some([lon, lat]) = c.geolocation {
                    v["core:geolocation"] = serde_json::json!({
                        "type": "Point",
                        "coordinates": [lon, lat],
                    });
                }
                v
            })
            .collect();

        let doc = serde_json::json!({
            "global": global,
            "captures": captures,
            "annotations": meta.annotations,
        });

        let mut meta_file = fs::File::create(&self.meta_path)
            .with_context(|| format!("create SigMF meta file {}", self.meta_path.display()))?;
        serde_json::to_writer_pretty(&mut meta_file, &doc).context("write SigMF meta JSON")?;
        meta_file.flush().context("flush SigMF meta file")
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

    /// Centre frequency in Hz of the recording's earliest capture — by
    /// `core:sample_start`, not by position in the JSON array — or `None`
    /// if that capture declares none. Where several captures share the
    /// earliest start the last listed describes it, as in playback.
    ///
    /// A later capture's frequency is never substituted: it describes
    /// other samples. Playback tags each capture with its own frequency.
    pub fn center_frequency_hz(&self) -> Option<f64> {
        let captures = self.captures_by_start();
        let first = captures.first()?.sample_start;
        captures
            .into_iter()
            .take_while(|c| c.sample_start == first)
            .last()?
            .frequency
    }

    /// The captures in `core:sample_start` order. The spec requires
    /// ascending order; this tolerates writers that don't. The sort is
    /// stable, so captures sharing a start keep their listed order.
    pub(crate) fn captures_by_start(&self) -> Vec<&Capture> {
        let mut captures: Vec<_> = self.captures.iter().collect();
        captures.sort_by_key(|c| c.sample_start);
        captures
    }

    /// Global sample rate in Hz. Required by this reader for playback.
    pub fn sample_rate_hz(&self) -> f64 {
        self.global.sample_rate
    }

    /// Resolved [`DataType`] or a spec-derived error.
    pub fn data_type(&self) -> Result<DataType> {
        DataType::from_spec(&self.global.datatype)
    }
}

// Append suffixes without treating dots in a recording basename as extensions
// or replacing non-UTF-8 path bytes. Only known SigMF suffixes are stripped.
fn pair_paths(path: &Path) -> (PathBuf, PathBuf) {
    let base = if path
        .extension()
        .is_some_and(|ext| ext == "sigmf-meta" || ext == "sigmf-data")
    {
        path.with_extension("")
    } else {
        path.to_path_buf()
    };
    let mut meta = base.as_os_str().to_os_string();
    meta.push(".sigmf-meta");
    let mut data = base.into_os_string();
    data.push(".sigmf-data");
    (PathBuf::from(meta), PathBuf::from(data))
}

/// Given a `.sigmf-meta`, `.sigmf-data`, or recording basename (which
/// may contain dots), return the `(meta, data)` pair.
pub fn resolve_pair(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let (meta, data) = pair_paths(path);
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
    if path
        .extension()
        .is_some_and(|ext| ext == "sigmf-meta" || ext == "sigmf-data")
    {
        return true;
    }
    let (meta, data) = pair_paths(path);
    meta.exists() && data.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn both_cf32_encoders_write_exact_little_endian_bits() {
        let dir = tempfile::tempdir().unwrap();
        let patterns = [
            0x3f80_0000,
            0x8000_0000,
            0x7f80_0000,
            0x7fc0_1234,
            0x0000_0001,
        ];
        // Cross the portable encoder's 64 KiB buffer boundary, including a tail.
        let samples: Vec<_> = (0..8193)
            .map(|i| {
                Complex32::new(
                    f32::from_bits(patterns[i % patterns.len()]),
                    f32::from_bits(patterns[(i + 1) % patterns.len()]),
                )
            })
            .collect();
        let expected: Vec<_> = samples
            .iter()
            .flat_map(|s| s.re.to_le_bytes().into_iter().chain(s.im.to_le_bytes()))
            .collect();
        for portable in [false, true] {
            let base = dir
                .path()
                .join(if portable { "portable" } else { "native" });
            let mut writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
            if portable {
                writer.write_cf32_le_portable(&[]).unwrap();
                writer.write_cf32_le_portable(&samples).unwrap();
            } else {
                writer.write_samples(&[]).unwrap();
                writer.write_samples(&samples).unwrap();
            }
            writer
                .finalize(SigmfWriterMeta {
                    sample_rate_hz: 1e6,
                    hardware: None,
                    description: None,
                    recorder: None,
                    captures: vec![],
                    annotations: vec![],
                })
                .unwrap();
            assert_eq!(
                std::fs::read(base.with_extension("sigmf-data")).unwrap(),
                expected
            );
        }
    }

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
    fn ci8_decode_uses_the_hackrf_full_scale() {
        let bytes = [
            127u8,        // re = 127
            -128i8 as u8, // im = -128
        ];
        let decoded = DataType::Ci8.decode(&bytes);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].re, 127.0 / 128.0);
        assert_eq!(decoded[0].im, -1.0);
    }

    /// Every integer code survives write → read, and a value between two
    /// codes lands on the nearer one rather than the one toward zero.
    #[test]
    fn integer_writers_round_to_nearest_and_round_trip_every_code() {
        let dir = tempfile::tempdir().unwrap();
        for (datatype, scale, codes) in [
            (DataType::Ci8, 128.0f32, (-128i32..=127).collect::<Vec<_>>()),
            (
                DataType::Ci16Le,
                32768.0f32,
                (-32768i32..=32767).collect::<Vec<_>>(),
            ),
        ] {
            let exact: Vec<_> = codes
                .iter()
                .map(|&c| Complex32::new(c as f32 / scale, -(c as f32) / scale))
                .collect();
            let base = dir.path().join(datatype.to_spec());
            let mut writer = SigmfWriter::create(&base, datatype).unwrap();
            writer.write_samples(&exact).unwrap();
            // 0.7 of a code above zero, and below: truncation gives 0, 0.
            writer
                .write_samples(&[Complex32::new(0.7 / scale, -0.7 / scale)])
                .unwrap();
            writer
                .finalize(SigmfWriterMeta {
                    sample_rate_hz: 1e6,
                    hardware: None,
                    description: None,
                    recorder: None,
                    captures: vec![],
                    annotations: vec![],
                })
                .unwrap();
            let decoded =
                datatype.decode(&std::fs::read(base.with_extension("sigmf-data")).unwrap());
            let (between, round_trip) = decoded.split_last().unwrap();
            for (got, want) in round_trip.iter().zip(&exact) {
                assert_eq!(got.re, want.re, "{datatype:?}");
                // -(-full scale) saturates to the largest positive code.
                assert_eq!(got.im, want.im.min((scale - 1.0) / scale), "{datatype:?}");
            }
            assert_eq!(*between, Complex32::new(1.0 / scale, -1.0 / scale));
        }
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

    /// The earliest capture by `core:sample_start` names the recording's
    /// frequency, whatever order the array lists them in; a tie goes to
    /// the last listed, as in playback; and an earliest capture with no
    /// frequency yields `None` rather than a later capture's.
    #[test]
    fn center_frequency_is_the_earliest_captures_not_the_first_listed() {
        let meta = |captures: &str| -> SigmfMetadata {
            serde_json::from_str(&format!(
                r#"{{"global": {{"core:datatype": "cf32_le", "core:sample_rate": 1}},
                    "captures": {captures}}}"#
            ))
            .unwrap()
        };
        let out_of_order = meta(
            r#"[{"core:sample_start": 100, "core:frequency": 2.0},
                {"core:sample_start": 0, "core:frequency": 1.0}]"#,
        );
        assert_eq!(out_of_order.center_frequency_hz(), Some(1.0));
        let tie = meta(
            r#"[{"core:sample_start": 0, "core:frequency": 5.0},
                {"core:sample_start": 0, "core:frequency": 6.0}]"#,
        );
        assert_eq!(tie.center_frequency_hz(), Some(6.0));
        let unstated = meta(
            r#"[{"core:sample_start": 100, "core:frequency": 2.0},
                {"core:sample_start": 0}]"#,
        );
        assert_eq!(unstated.center_frequency_hz(), None);
    }

    /// NaN has no integer code; `as` would have written it as 0 (and an
    /// infinity as full scale), recording a sample nobody received.
    #[test]
    fn integer_writers_refuse_non_finite_samples_and_write_nothing() {
        let dir = tempfile::tempdir().unwrap();
        for datatype in [DataType::Ci8, DataType::Ci16Le] {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let base = dir.path().join(format!("{}_{bad}", datatype.to_spec()));
                let mut writer = SigmfWriter::create(&base, datatype).unwrap();
                let err = writer
                    .write_samples(&[Complex32::new(0.5, 0.5), Complex32::new(0.1, bad)])
                    .unwrap_err();
                assert!(err.to_string().contains("sample 1"), "{err}");
                drop(writer);
                assert_eq!(
                    std::fs::metadata(base.with_extension("sigmf-data"))
                        .unwrap()
                        .len(),
                    0,
                    "{datatype:?} {bad}: a refused call must write nothing"
                );
            }
        }
        // cf32 represents NaN, so it is written as given.
        let base = dir.path().join("cf32");
        let mut writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
        writer
            .write_samples(&[Complex32::new(f32::NAN, 0.0)])
            .unwrap();
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
    fn parse_metadata_missing_version_still_loads() {
        // core:version is spec-mandatory, but some real-world writers
        // omit it. The capture should still load rather than be dropped.
        let body = r#"{
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1000000
            },
            "captures": [{ "core:sample_start": 0 }]
        }"#;
        let meta: SigmfMetadata = serde_json::from_str(body).unwrap();
        assert_eq!(meta.sample_rate_hz(), 1_000_000.0);
        assert_eq!(meta.global.version, "");
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
    fn sigmf_writer_round_trips_cf32_le() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("rec");
        let samples = [Complex32::new(1.5, -2.5), Complex32::new(0.25, 0.5)];

        let mut writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
        writer.write_samples(&samples).unwrap();
        writer
            .finalize(SigmfWriterMeta {
                sample_rate_hz: 2_000_000.0,
                hardware: Some("test-hw".into()),
                description: Some("test capture".into()),
                recorder: Some("test-recorder".into()),
                captures: vec![SigmfCapture {
                    sample_start: 0,
                    frequency_hz: Some(915_000_000.0),
                    datetime_rfc3339: Some("2026-01-01T00:00:00Z".into()),
                    geolocation: Some([-122.4, 37.8]),
                }],
                annotations: vec![],
            })
            .unwrap();

        let meta = SigmfMetadata::load(&base.with_extension("sigmf-meta")).unwrap();
        assert_eq!(meta.data_type().unwrap(), DataType::Cf32Le);
        assert_eq!(meta.sample_rate_hz(), 2_000_000.0);
        assert_eq!(meta.center_frequency_hz(), Some(915_000_000.0));
        assert_eq!(meta.global.hardware.as_deref(), Some("test-hw"));

        let bytes = std::fs::read(base.with_extension("sigmf-data")).unwrap();
        let decoded = DataType::Cf32Le.decode(&bytes);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0].re - 1.5).abs() < 1e-6);
        assert!((decoded[1].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmf_writer_round_trips_ci8() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("rec8");
        let samples = [Complex32::new(1.0, -1.0)];

        let mut writer = SigmfWriter::create(&base, DataType::Ci8).unwrap();
        writer.write_samples(&samples).unwrap();
        writer
            .finalize(SigmfWriterMeta {
                sample_rate_hz: 20_000_000.0,
                hardware: None,
                description: None,
                recorder: None,
                captures: vec![],
                annotations: vec![],
            })
            .unwrap();

        let meta = SigmfMetadata::load(&base.with_extension("sigmf-meta")).unwrap();
        assert_eq!(meta.data_type().unwrap(), DataType::Ci8);
        assert!(meta.center_frequency_hz().is_none());

        let bytes = std::fs::read(base.with_extension("sigmf-data")).unwrap();
        assert_eq!(bytes.len(), 2);
        let decoded = DataType::Ci8.decode(&bytes);
        assert!((decoded[0].re - 1.0).abs() < 0.05);
        assert!((decoded[0].im + 1.0).abs() < 0.05);
    }

    #[test]
    fn sigmf_writer_write_raw_passes_bytes_through() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("raw");
        let raw = [1u8, 2, 3, 4, 5, 6, 7, 8];

        let mut writer = SigmfWriter::create(&base, DataType::Ci8).unwrap();
        writer.write_raw(&raw).unwrap();
        writer
            .finalize(SigmfWriterMeta {
                sample_rate_hz: 1_000_000.0,
                hardware: None,
                description: None,
                recorder: None,
                captures: vec![],
                annotations: vec![],
            })
            .unwrap();

        let bytes = std::fs::read(base.with_extension("sigmf-data")).unwrap();
        assert_eq!(bytes, raw);
    }

    #[test]
    fn sigmf_writer_passes_through_annotations() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("annotated");

        let mut writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
        writer.write_raw(&[]).unwrap();
        writer
            .finalize(SigmfWriterMeta {
                sample_rate_hz: 1_000_000.0,
                hardware: None,
                description: None,
                recorder: None,
                captures: vec![],
                annotations: vec![serde_json::json!({
                    "core:sample_start": 0,
                    "core:sample_count": 100,
                    "core:label": "test-signal"
                })],
            })
            .unwrap();

        let body = std::fs::read_to_string(base.with_extension("sigmf-meta")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(doc["annotations"][0]["core:label"], "test-signal");
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

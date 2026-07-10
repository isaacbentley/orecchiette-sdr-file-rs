#![doc = include_str!("../README.md")]
//! File-backed SdrSource implementations for SDR applications.
//!
//! Two source types share the same `SdrSource` trait:
//!
//! * [`RawIqFileSource`] — header-less IQ recordings, used for the
//!   orchestrator's legacy `.bin` (`i16` scaled by `1/32768`) and
//!   raw-`f32` capture formats. Centre frequency is supplied by the
//!   caller because the file itself has no metadata.
//! * [`SigmfFileSource`] — [SigMF](https://github.com/sigmf/sigmf-spec)
//!   recordings paired as `.sigmf-meta` (JSON) plus `.sigmf-data`
//!   (raw payload). Centre frequency, sample rate, and datatype all
//!   come from the metadata; the caller doesn't need to know.
//!
//! Both carry no notion of channel hopping or dwell — the file *is*
//! the capture, played back at its natural rate. Adaptive-dwell
//! input is accepted at the trait boundary but unused.

pub mod sigmf;

use crossbeam::channel;
use num_complex::Complex32;
use orecchiette_sdr_source_rs::{
    DwellAdvice, IqPacket, SdrError, SdrHandle, SdrSource, SourceConfig,
};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use tracing::{info, warn};

pub use sigmf::{DataType as SigmfDataType, SigmfMetadata, looks_like_sigmf};

const IO_BUFFER_BYTES: usize = 1024 * 1024;
const PACKET_SAMPLES: usize = 1_048_576;

/// Raw-IQ file source. Accepts one or more pre-globbed paths and
/// streams them sequentially.
pub struct RawIqFileSource {
    pub paths: Vec<PathBuf>,
    /// Center frequency tagged on every emitted [`IqPacket`]. Files
    /// don't carry frequency metadata — the caller passes it in from
    /// the CLI (or whichever record their capture provenance).
    pub center_frequency_hz: f64,
}

impl SdrSource for RawIqFileSource {
    fn start(
        self: Box<Self>,
        config: SourceConfig,
        _advice: Arc<dyn DwellAdvice>,
    ) -> Result<SdrHandle, SdrError> {
        if self.paths.is_empty() {
            return Err(SdrError::BadConfig(
                "RawIqFileSource: no paths to play".into(),
            ));
        }
        if config.sample_rate_hz <= 0.0 {
            return Err(SdrError::BadConfig("RawIqFileSource: sample_rate_hz must be > 0".into()));
        }
        let (tx, receiver) = channel::bounded::<IqPacket>(1024);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop_flag.clone();
        let paths = self.paths.clone();
        let center = self.center_frequency_hz;
        let rate = config.sample_rate_hz as f32;

        let (pool_tx, pool_rx) = channel::bounded::<Vec<Complex32>>(1024);
        for _ in 0..1024 {
            let _ = pool_tx.send(Vec::with_capacity(PACKET_SAMPLES));
        }

        let capture_thread = thread::spawn(move || {
            let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                if let Err(e) = (move || -> Result<(), anyhow::Error> {
                    for path in paths {
                        if stop_for_thread.load(Ordering::SeqCst) {
                            break;
                        }
                        let is_bin = path.extension().is_some_and(|e| e == "bin");
                        let mut file = match File::open(&path) {
                            Ok(f) => f,
                            Err(e) => {
                                warn!(
                                    "orecchiette-sdr-file: failed to open {}: {e}",
                                    path.display()
                                );
                                continue;
                            }
                        };
                        let mut buffer = vec![0u8; IO_BUFFER_BYTES];
                        let mut leftovers: Vec<Complex32> = Vec::new();

                        // Bytes 0..pending hold a partial sample carried over
                        // from the previous read. Carrying in the buffer (rather
                        // than seeking back) keeps IQ alignment across short
                        // reads without re-reading: a seek-back of a tail
                        // shorter than one sample re-reads the same bytes
                        // forever on truncated files.
                        let mut pending = 0usize;
                        loop {
                            if stop_for_thread.load(Ordering::SeqCst) {
                                return Ok(());
                            }
                            let n = file.read(&mut buffer[pending..])?;
                            if n == 0 {
                                // EOF: flush any partial-sample tail before moving
                                // on to the next file. Tail packets are smaller
                                // than `PACKET_SAMPLES`; the downstream DSP gates
                                // on a minimum buffer size, so a very short tail
                                // simply doesn't trigger a detection. Dropping it
                                // silently (the pre-fix behaviour) lost the last
                                // < ~17 ms of every file at 15.36 MSPS.
                                if pending > 0 {
                                    warn!(
                                        "orecchiette-sdr-file: {} ends in {} byte(s) of a truncated sample; discarded",
                                        path.display(),
                                        pending
                                    );
                                }
                                if !leftovers.is_empty() {
                                    let mut pooled = pool_rx
                                        .try_recv()
                                        .unwrap_or_else(|_| Vec::with_capacity(PACKET_SAMPLES));
                                    pooled.clear();
                                    pooled.extend_from_slice(&leftovers);
                                    leftovers.clear();
                                    let pkt = IqPacket {
                                        samples: orecchiette_sdr_source_rs::PooledIqBuffer::new_pooled(
                                            pooled,
                                            pool_tx.clone(),
                                        ),
                                        center_frequency_hz: center,
                                        sample_rate_hz: rate,
                                        overrun: false,
                                    };
                                    if tx.send(pkt).is_err() {
                                        return Ok(());
                                    }
                                }
                                break;
                            }
                            let bps = if is_bin { 4 } else { 8 };
                            let avail = pending + n;
                            let full_bytes = avail - (avail % bps);
                            let mut samples = decode_block(&buffer[..full_bytes], is_bin);
                            // Carry the partial-sample tail to the front for the
                            // next read.
                            buffer.copy_within(full_bytes..avail, 0);
                            pending = avail - full_bytes;

                            let mut joined = Vec::with_capacity(leftovers.len() + samples.len());
                            joined.append(&mut leftovers);
                            joined.append(&mut samples);

                            for chunk in joined.chunks(PACKET_SAMPLES) {
                                if chunk.len() < PACKET_SAMPLES {
                                    leftovers.extend_from_slice(chunk);
                                    break;
                                }
                                let mut pooled = pool_rx
                                    .try_recv()
                                    .unwrap_or_else(|_| Vec::with_capacity(PACKET_SAMPLES));
                                pooled.clear();
                                pooled.extend_from_slice(chunk);
                                let pkt = IqPacket {
                                    samples: orecchiette_sdr_source_rs::PooledIqBuffer::new_pooled(
                                        pooled,
                                        pool_tx.clone(),
                                    ),
                                    center_frequency_hz: center,
                                    sample_rate_hz: rate,
                                    overrun: false,
                                };
                                if tx.send(pkt).is_err() {
                                    return Ok(()); // consumer dropped
                                }
                            }
                        }
                    }
                    Ok(())
                })() {
                    tracing::error!("[file] Capture thread failed: {:?}", e);
                }
            }));
            if let Err(e) = panic_res {
                tracing::error!("[file] Capture thread panicked: {:?}", e);
            }
        });

        let stop_handle = stop_flag.clone();
        let stop = Box::new(move || stop_handle.store(true, Ordering::SeqCst));
        let wait = Box::new(move || {
            if let Err(e) = capture_thread.join() {
                tracing::error!("[file] capture thread join failed: {:?}", e);
            }
        });
        Ok(SdrHandle {
            receiver,
            stop,
            wait,
        })
    }
}

/// SigMF file source. Accepts one or more `.sigmf-meta` / `.sigmf-data`
/// paths (or the bare recording basename); plays them sequentially,
/// pulling sample rate + centre frequency + datatype from each
/// recording's metadata.
///
/// Use this in preference to [`RawIqFileSource`] whenever the capture
/// ships a `.sigmf-meta` sidecar — the metadata is the source of
/// truth for centre frequency, and `IqPacket`s are tagged accordingly.
pub struct SigmfFileSource {
    /// Each path is either a `.sigmf-meta`, a `.sigmf-data`, or a
    /// bare recording basename (no extension) whose `.sigmf-meta` and
    /// `.sigmf-data` siblings both exist.
    pub paths: Vec<PathBuf>,
}

impl SdrSource for SigmfFileSource {
    fn start(
        self: Box<Self>,
        _config: SourceConfig,
        _advice: Arc<dyn DwellAdvice>,
    ) -> Result<SdrHandle, SdrError> {
        if self.paths.is_empty() {
            return Err(SdrError::BadConfig(
                "SigmfFileSource: no paths to play".into(),
            ));
        }
        let (tx, receiver) = channel::bounded::<IqPacket>(1024);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop_flag.clone();
        let paths = self.paths.clone();

        let (pool_tx, pool_rx) = channel::bounded::<Vec<Complex32>>(1024);
        for _ in 0..1024 {
            let _ = pool_tx.send(Vec::with_capacity(PACKET_SAMPLES));
        }

        let capture_thread = thread::spawn(move || {
            let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                if let Err(e) = (move || -> Result<(), anyhow::Error> {
                    for path in paths {
                        if stop_for_thread.load(Ordering::SeqCst) {
                            break;
                        }
                        let (meta_path, data_path) = match sigmf::resolve_pair(&path) {
                            Ok(pair) => pair,
                            Err(e) => {
                                warn!("sigmf: could not resolve {}: {e}", path.display());
                                continue;
                            }
                        };
                        let meta = match SigmfMetadata::load(&meta_path) {
                            Ok(m) => m,
                            Err(e) => {
                                warn!("sigmf: bad metadata {}: {e}", meta_path.display());
                                continue;
                            }
                        };
                        let datatype = match meta.data_type() {
                            Ok(d) => d,
                            Err(e) => {
                                warn!("sigmf: {}: {e}", meta_path.display());
                                continue;
                            }
                        };
                        let center_hz = meta.center_frequency_hz().unwrap_or_else(|| {
                            warn!(
                                "sigmf: {} has no core:frequency in any capture; tagging packets with 0 Hz",
                                meta_path.display()
                            );
                            0.0
                        });
                        let sample_rate = meta.sample_rate_hz();
                        if sample_rate <= 0.0 {
                            warn!("sigmf: invalid sample rate {} Hz in {}; skipping file", sample_rate, meta_path.display());
                            continue;
                        }
                        let sample_rate_f32 = sample_rate as f32;
                        info!(
                            "sigmf: playing {} ({}, {} MHz @ {:.3} MSPS)",
                            data_path.display(),
                            meta.global.datatype,
                            center_hz / 1e6,
                            sample_rate / 1e6,
                        );

                        let mut file = match File::open(&data_path) {
                            Ok(f) => f,
                            Err(e) => {
                                warn!("sigmf: failed to open {}: {e}", data_path.display());
                                continue;
                            }
                        };
                        let mut buffer = vec![0u8; IO_BUFFER_BYTES];
                        let mut leftovers: Vec<Complex32> = Vec::new();

                        // Partial-sample carry — see the matching note in
                        // `RawIqFileSource::start`.
                        let mut pending = 0usize;
                        loop {
                            if stop_for_thread.load(Ordering::SeqCst) {
                                return Ok(());
                            }
                            let n = file.read(&mut buffer[pending..])?;
                            if n == 0 {
                                // EOF: flush any partial-sample tail (see the
                                // matching note in `RawIqFileSource::start`).
                                if pending > 0 {
                                    warn!(
                                        "sigmf: {} ends in {} byte(s) of a truncated sample; discarded",
                                        data_path.display(),
                                        pending
                                    );
                                }
                                if !leftovers.is_empty() {
                                    let mut pooled = pool_rx
                                        .try_recv()
                                        .unwrap_or_else(|_| Vec::with_capacity(PACKET_SAMPLES));
                                    pooled.clear();
                                    pooled.extend_from_slice(&leftovers);
                                    leftovers.clear();
                                    let pkt = IqPacket {
                                        samples: orecchiette_sdr_source_rs::PooledIqBuffer::new_pooled(
                                            pooled,
                                            pool_tx.clone(),
                                        ),
                                        center_frequency_hz: center_hz,
                                        sample_rate_hz: sample_rate_f32,
                                        overrun: false,
                                    };
                                    if tx.send(pkt).is_err() {
                                        return Ok(());
                                    }
                                }
                                break;
                            }
                            // Round down to a multiple of `bytes_per_sample` so
                            // we never split an IQ pair across reads; the
                            // trailing partial bytes are carried in the front of
                            // the buffer for the next read.
                            let bps = datatype.bytes_per_sample();
                            let avail = pending + n;
                            let full_bytes = avail - (avail % bps);
                            let mut samples = datatype.decode(&buffer[..full_bytes]);
                            buffer.copy_within(full_bytes..avail, 0);
                            pending = avail - full_bytes;

                            let mut joined = Vec::with_capacity(leftovers.len() + samples.len());
                            joined.append(&mut leftovers);
                            joined.append(&mut samples);

                            for chunk in joined.chunks(PACKET_SAMPLES) {
                                if chunk.len() < PACKET_SAMPLES {
                                    leftovers.extend_from_slice(chunk);
                                    break;
                                }
                                let mut pooled = pool_rx
                                    .try_recv()
                                    .unwrap_or_else(|_| Vec::with_capacity(PACKET_SAMPLES));
                                pooled.clear();
                                pooled.extend_from_slice(chunk);
                                let pkt = IqPacket {
                                    samples: orecchiette_sdr_source_rs::PooledIqBuffer::new_pooled(
                                        pooled,
                                        pool_tx.clone(),
                                    ),
                                    center_frequency_hz: center_hz,
                                    sample_rate_hz: sample_rate_f32,
                                    overrun: false,
                                };
                                if tx.send(pkt).is_err() {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    Ok(())
                })() {
                    tracing::error!("[file] Capture thread failed: {:?}", e);
                }
            }));
            if let Err(e) = panic_res {
                tracing::error!("[file] Capture thread panicked: {:?}", e);
            }
        });

        let stop_handle = stop_flag.clone();
        let stop = Box::new(move || stop_handle.store(true, Ordering::SeqCst));
        let wait = Box::new(move || {
            if let Err(e) = capture_thread.join() {
                tracing::error!("[file] capture thread join failed: {:?}", e);
            }
        });
        Ok(SdrHandle {
            receiver,
            stop,
            wait,
        })
    }
}

/// Decode a byte slice into `Complex32` samples per the format the
/// file extension implies. `.bin` is int16 scaled by 1/32768; anything
/// else is interleaved `f32`.
fn decode_block(bytes: &[u8], is_bin: bool) -> Vec<Complex32> {
    if is_bin {
        bytes
            .chunks_exact(4)
            .map(|c| {
                let re = i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
                let im = i16::from_le_bytes([c[2], c[3]]) as f32 / 32768.0;
                Complex32::new(re, im)
            })
            .collect()
    } else {
        bytes
            .chunks_exact(8)
            .map(|c| {
                let re = f32::from_le_bytes(c[0..4].try_into().unwrap());
                let im = f32::from_le_bytes(c[4..8].try_into().unwrap());
                Complex32::new(re, im)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_block_f32_round_trip() {
        // Encode two IQ pairs as little-endian f32, then decode.
        let samples = [Complex32::new(1.5, -2.5), Complex32::new(0.25, 0.5)];
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for s in samples {
            bytes.extend_from_slice(&s.re.to_le_bytes());
            bytes.extend_from_slice(&s.im.to_le_bytes());
        }
        let decoded = decode_block(&bytes, false);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0].re - 1.5).abs() < 1e-6);
        assert!((decoded[0].im + 2.5).abs() < 1e-6);
        assert!((decoded[1].re - 0.25).abs() < 1e-6);
        assert!((decoded[1].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn decode_block_bin_scales_int16() {
        // int16 32767 → ~1.0, -32768 → ~-1.0, scaled by 1/32768.
        let bytes = [
            0xFF, 0x7F, // re = 32767
            0x00, 0x80, // im = -32768
        ];
        let decoded = decode_block(&bytes, true);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0].re - (32767.0 / 32768.0)).abs() < 1e-6);
        assert!((decoded[0].im + 1.0).abs() < 1e-6);
    }

    /// End-to-end: write a synthetic .sigmf-meta + .sigmf-data pair,
    /// hand the path to `SigmfFileSource::start`, drain the receiver,
    /// and verify the centre frequency + decoded samples come from
    /// the metadata.
    #[test]
    fn sigmf_file_source_round_trip() {
        use std::sync::Arc;
        use std::time::Duration;

        // Build N IQ pairs as raw cf32_le bytes.
        let n_packets = 3;
        let total_samples = PACKET_SAMPLES * n_packets;
        let mut data_bytes = Vec::with_capacity(total_samples * 8);
        for i in 0..total_samples {
            let re = (i as f32) * 1.0e-6;
            let im = (i as f32) * -2.0e-6;
            data_bytes.extend_from_slice(&re.to_le_bytes());
            data_bytes.extend_from_slice(&im.to_le_bytes());
        }

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("synthetic");
        let meta_path = PathBuf::from(format!("{}.sigmf-meta", base.display()));
        let data_path = PathBuf::from(format!("{}.sigmf-data", base.display()));
        std::fs::write(
            &meta_path,
            r#"{
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": 1000000,
                    "core:version": "1.0.0"
                },
                "captures": [
                    { "core:sample_start": 0, "core:frequency": 2435000000 }
                ]
            }"#,
        )
        .unwrap();
        std::fs::write(&data_path, &data_bytes).unwrap();

        struct NoSignal;
        impl DwellAdvice for NoSignal {
            fn latest_signal_at(&self, _: u64) -> Option<std::time::Instant> {
                None
            }
        }
        let advice: Arc<dyn DwellAdvice> = Arc::new(NoSignal);
        let config = SourceConfig {
            sample_rate_hz: 0.0,
            channels_hz: vec![],
            dwell_min: Duration::from_millis(0),
            dwell_max: Duration::from_millis(0),
            dwell_extension: Duration::from_millis(0),
        };
        let source = Box::new(SigmfFileSource {
            paths: vec![meta_path],
        });
        let handle = source.start(config, advice).expect("start");

        let mut received = 0;
        let mut last_re = -1.0f32;
        while let Ok(pkt) = handle.receiver.recv_timeout(Duration::from_secs(2)) {
            assert_eq!(pkt.samples.len(), PACKET_SAMPLES);
            // Centre freq + sample rate come from the metadata, not
            // any CLI arg or SourceConfig.
            assert!((pkt.center_frequency_hz - 2_435_000_000.0).abs() < 1.0);
            assert!((pkt.sample_rate_hz - 1_000_000.0).abs() < 1.0);
            // Monotonic re — confirms samples arrive in order.
            assert!(pkt.samples[0].re >= last_re);
            last_re = pkt.samples.last().unwrap().re;
            received += 1;
        }
        assert_eq!(received, n_packets, "expected {n_packets} packets");
    }

    /// Raw `.cf32` whose sample count isn't a multiple of
    /// `PACKET_SAMPLES` must emit the tail as a final partial packet
    /// (same fix as `sigmf_file_source_emits_partial_tail`, mirrored
    /// to the raw backend).
    #[test]
    fn raw_iq_file_source_emits_partial_tail() {
        use std::sync::Arc;
        use std::time::Duration;

        let tail = 4321_usize;
        let total_samples = PACKET_SAMPLES + tail;
        let mut data_bytes = Vec::with_capacity(total_samples * 8);
        for i in 0..total_samples {
            let re = (i as f32) * 1.0e-6;
            let im = (i as f32) * -2.0e-6;
            data_bytes.extend_from_slice(&re.to_le_bytes());
            data_bytes.extend_from_slice(&im.to_le_bytes());
        }

        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("tail.cf32");
        std::fs::write(&data_path, &data_bytes).unwrap();

        struct NoSignal;
        impl DwellAdvice for NoSignal {
            fn latest_signal_at(&self, _: u64) -> Option<std::time::Instant> {
                None
            }
        }
        let advice: Arc<dyn DwellAdvice> = Arc::new(NoSignal);
        let config = SourceConfig {
            sample_rate_hz: 15_360_000.0,
            channels_hz: vec![],
            dwell_min: Duration::from_millis(0),
            dwell_max: Duration::from_millis(0),
            dwell_extension: Duration::from_millis(0),
        };
        let source = Box::new(RawIqFileSource {
            paths: vec![data_path],
            center_frequency_hz: 2_435_000_000.0,
        });
        let handle = source.start(config, advice).expect("start");

        let mut packets = Vec::new();
        while let Ok(pkt) = handle.receiver.recv_timeout(Duration::from_secs(2)) {
            packets.push(pkt);
        }
        assert_eq!(packets.len(), 2, "1 full packet + 1 partial tail");
        assert_eq!(packets[0].samples.len(), PACKET_SAMPLES);
        assert_eq!(packets[1].samples.len(), tail, "tail size preserved");
    }

    /// A file truncated mid-sample (trailing bytes < one IQ pair) must
    /// terminate with the truncated bytes discarded — not loop forever.
    /// Pre-fix: the seek-back rewound exactly the short tail read, so
    /// the reader re-read the same bytes endlessly and the source never
    /// reached EOF.
    #[test]
    fn raw_iq_file_source_terminates_on_truncated_sample() {
        use std::sync::Arc;
        use std::time::Duration;

        let tail = 123_usize;
        let total_samples = PACKET_SAMPLES + tail;
        let mut data_bytes = Vec::with_capacity(total_samples * 8 + 3);
        for i in 0..total_samples {
            let re = (i as f32) * 1.0e-6;
            let im = (i as f32) * -2.0e-6;
            data_bytes.extend_from_slice(&re.to_le_bytes());
            data_bytes.extend_from_slice(&im.to_le_bytes());
        }
        // Truncated sample: 3 stray bytes that can't form an IQ pair.
        data_bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);

        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("truncated.cf32");
        std::fs::write(&data_path, &data_bytes).unwrap();

        struct NoSignal;
        impl DwellAdvice for NoSignal {
            fn latest_signal_at(&self, _: u64) -> Option<std::time::Instant> {
                None
            }
        }
        let advice: Arc<dyn DwellAdvice> = Arc::new(NoSignal);
        let config = SourceConfig {
            sample_rate_hz: 15_360_000.0,
            channels_hz: vec![],
            dwell_min: Duration::from_millis(0),
            dwell_max: Duration::from_millis(0),
            dwell_extension: Duration::from_millis(0),
        };
        let source = Box::new(RawIqFileSource {
            paths: vec![data_path],
            center_frequency_hz: 2_435_000_000.0,
        });
        let handle = source.start(config, advice).expect("start");

        let mut packets = Vec::new();
        while let Ok(pkt) = handle.receiver.recv_timeout(Duration::from_secs(2)) {
            packets.push(pkt);
        }
        assert_eq!(
            packets.len(),
            2,
            "1 full packet + 1 partial tail (truncated bytes discarded)"
        );
        assert_eq!(packets[0].samples.len(), PACKET_SAMPLES);
        assert_eq!(packets[1].samples.len(), tail);
    }

    /// Files whose sample count isn't a multiple of `PACKET_SAMPLES`
    /// must emit the tail as a final partial packet rather than
    /// silently dropping it on EOF. Pre-fix: the loop broke on
    /// `read == 0` and discarded everything still sitting in
    /// `leftovers`.
    #[test]
    fn sigmf_file_source_emits_partial_tail() {
        use std::sync::Arc;
        use std::time::Duration;

        let tail = 1234_usize;
        let total_samples = PACKET_SAMPLES + tail;
        let mut data_bytes = Vec::with_capacity(total_samples * 8);
        for i in 0..total_samples {
            let re = (i as f32) * 1.0e-6;
            let im = (i as f32) * -2.0e-6;
            data_bytes.extend_from_slice(&re.to_le_bytes());
            data_bytes.extend_from_slice(&im.to_le_bytes());
        }

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("with_tail");
        let meta_path = PathBuf::from(format!("{}.sigmf-meta", base.display()));
        let data_path = PathBuf::from(format!("{}.sigmf-data", base.display()));
        std::fs::write(
            &meta_path,
            r#"{
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": 1000000,
                    "core:version": "1.0.0"
                },
                "captures": [
                    { "core:sample_start": 0, "core:frequency": 2435000000 }
                ]
            }"#,
        )
        .unwrap();
        std::fs::write(&data_path, &data_bytes).unwrap();

        struct NoSignal;
        impl DwellAdvice for NoSignal {
            fn latest_signal_at(&self, _: u64) -> Option<std::time::Instant> {
                None
            }
        }
        let advice: Arc<dyn DwellAdvice> = Arc::new(NoSignal);
        let config = SourceConfig {
            sample_rate_hz: 0.0,
            channels_hz: vec![],
            dwell_min: Duration::from_millis(0),
            dwell_max: Duration::from_millis(0),
            dwell_extension: Duration::from_millis(0),
        };
        let source = Box::new(SigmfFileSource {
            paths: vec![meta_path],
        });
        let handle = source.start(config, advice).expect("start");

        let mut packets = Vec::new();
        while let Ok(pkt) = handle.receiver.recv_timeout(Duration::from_secs(2)) {
            packets.push(pkt);
        }
        assert_eq!(packets.len(), 2, "1 full packet + 1 partial tail");
        assert_eq!(packets[0].samples.len(), PACKET_SAMPLES);
        assert_eq!(packets[1].samples.len(), tail, "tail size preserved");
    }
}

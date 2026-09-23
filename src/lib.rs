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
//!   come from the metadata; the caller doesn't need to know. Each
//!   capture segment is tagged with its own `core:frequency`, and
//!   packets are cut at every `core:sample_start` so no packet spans
//!   two captures.
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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use tracing::{info, warn};

pub use sigmf::{DataType as SigmfDataType, SigmfMetadata, looks_like_sigmf};

const IO_BUFFER_BYTES: usize = 1024 * 1024;
const PACKET_SAMPLES: usize = 1_048_576;
// At most 64 MiB of full packets queued and 64 MiB of recycled buffers.
// Sample storage is allocated lazily; consumers own any packets they retain.
const PACKET_QUEUE_CAPACITY: usize = 8;

fn packet_sample_rate(rate: f64) -> Result<f32, &'static str> {
    let packet_rate = rate as f32;
    if !rate.is_finite() || !packet_rate.is_finite() || packet_rate <= 0.0 {
        return Err(
            "sample_rate_hz must be finite, positive, and representable as a positive finite f32",
        );
    }
    Ok(packet_rate)
}

/// The `center_frequency_hz` tag on samples whose centre frequency the
/// recording does not state: a SigMF capture without `core:frequency`,
/// or a stretch before the first capture.
///
/// Not a frequency — no receiver tunes to DC — and deliberately not one
/// borrowed from another capture, which describes other samples. A
/// consumer that knows the frequency from elsewhere (an operator's
/// `--center-freq`, say) supplies it; one that doesn't should treat a
/// non-positive centre as unknown rather than re-base on it.
pub const UNKNOWN_CENTER_HZ: f64 = 0.0;

/// A stretch of a data file that shares one centre frequency: a SigMF
/// capture segment, or the whole of a raw file. `start` is the absolute
/// sample index (not byte offset) where it begins.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Segment {
    start: u64,
    center_hz: f64,
}

/// What the next emitted packet is labelled with.
struct Tag {
    center_hz: f64,
    sample_rate_hz: f32,
    /// Carried on the next packet only, then cleared.
    overrun: bool,
}

/// Accumulates decoded samples into `PACKET_SAMPLES`-sized packets.
///
/// Samples are appended once into a single accumulator, reserved up
/// front for as much of the packet as the file can still supply, and
/// the accumulator itself becomes the packet (`mem::take`). The former
/// loop re-copied its partial-packet leftovers on every 1 MiB read, so
/// each sample was copied up to eight times before it left.
struct Packetizer {
    acc: Vec<Complex32>,
    tx: channel::Sender<IqPacket>,
    pool_tx: channel::Sender<Vec<Complex32>>,
    pool_rx: channel::Receiver<Vec<Complex32>>,
}

impl Packetizer {
    /// Append `samples`, emitting every packet that fills. `expected` is
    /// how many samples the current segment still holds, counting these,
    /// as far as the file length says; it only sizes the reservation.
    /// Returns `false` once the consumer has gone.
    fn push(&mut self, mut samples: &[Complex32], expected: u64, tag: &mut Tag) -> bool {
        let mut expected = expected.max(samples.len() as u64);
        while !samples.is_empty() {
            if self.acc.capacity() == 0 {
                // Lazily sized: a tiny recording must not reserve a full
                // 8 MiB packet it will never fill.
                let want = expected.min(PACKET_SAMPLES as u64) as usize;
                let mut pooled = self.pool_rx.try_recv().unwrap_or_default();
                pooled.clear();
                pooled.reserve_exact(want);
                self.acc = pooled;
            }
            let take = (PACKET_SAMPLES - self.acc.len()).min(samples.len());
            self.acc.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            expected = expected.saturating_sub(take as u64);
            if self.acc.len() == PACKET_SAMPLES && !self.emit(tag) {
                return false;
            }
        }
        true
    }

    /// Emit whatever has accumulated, however short. Used at file ends
    /// and capture boundaries.
    fn flush(&mut self, tag: &mut Tag) -> bool {
        self.acc.is_empty() || self.emit(tag)
    }

    fn emit(&mut self, tag: &mut Tag) -> bool {
        let pkt = IqPacket {
            samples: orecchiette_sdr_source_rs::PooledIqBuffer::new_pooled(
                std::mem::take(&mut self.acc),
                self.pool_tx.clone(),
            ),
            center_frequency_hz: tag.center_hz,
            sample_rate_hz: tag.sample_rate_hz,
            overrun: std::mem::take(&mut tag.overrun),
        };
        self.tx.send(pkt).is_ok()
    }
}

/// Stream one data file through `packetizer`, cutting packets at every
/// segment start. `segments` must be non-empty, begin at sample 0, and
/// have strictly increasing starts (see [`sigmf_segments`]).
///
/// Returns `Ok(false)` when playback should end altogether (stop
/// requested or consumer gone), `Ok(true)` at the file's end.
fn play_file(
    file: &mut File,
    path: &Path,
    datatype: sigmf::DataType,
    segments: &[Segment],
    sample_rate_hz: f32,
    packetizer: &mut Packetizer,
    stop: &AtomicBool,
) -> anyhow::Result<bool> {
    let bps = datatype.bytes_per_sample();
    // Only sizes reservations; a file still being written just grows the
    // accumulator the ordinary way.
    let file_samples = file.metadata().map_or(0, |m| m.len() / bps as u64);
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    let mut decoded: Vec<Complex32> = Vec::new();
    let mut segment = 0usize;
    let mut position = 0u64;
    let mut tag = Tag {
        center_hz: segments[0].center_hz,
        sample_rate_hz,
        overrun: false,
    };

    // Bytes 0..pending hold a partial sample carried over from the
    // previous read. Carrying in the buffer (rather than seeking back)
    // keeps IQ alignment across short reads without re-reading: a
    // seek-back of a tail shorter than one sample re-reads the same bytes
    // forever on truncated files.
    let mut pending = 0usize;
    loop {
        if stop.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let n = file.read(&mut buffer[pending..])?;
        if n == 0 {
            // EOF: flush the partial-packet tail. Tail packets are smaller
            // than `PACKET_SAMPLES`; the downstream DSP gates on a minimum
            // buffer size, so a very short tail simply doesn't trigger a
            // detection. Dropping it silently (the pre-fix behaviour) lost
            // the last < ~17 ms of every file at 15.36 MSPS.
            if pending > 0 {
                warn!(
                    "orecchiette-sdr-file: {} ends in {} byte(s) of a truncated sample; discarded",
                    path.display(),
                    pending
                );
            }
            return Ok(packetizer.flush(&mut tag));
        }
        // Round down to a whole number of IQ pairs; the trailing partial
        // bytes are carried to the front of the buffer for the next read.
        let avail = pending + n;
        let full_bytes = avail - (avail % bps);
        decoded.clear();
        datatype.decode_into(&buffer[..full_bytes], &mut decoded);
        buffer.copy_within(full_bytes..avail, 0);
        pending = avail - full_bytes;

        let mut rest = &decoded[..];
        while !rest.is_empty() {
            if let Some(next) = segments.get(segment + 1)
                && next.start <= position
            {
                // A capture boundary. Nothing either side of it may share a
                // packet, and a boundary that keeps the frequency is still a
                // break in the stream (SigMF starts a new capture exactly
                // when something about the recording is discontinuous), so
                // it is flagged the way a live source flags lost samples.
                if !packetizer.flush(&mut tag) {
                    return Ok(false);
                }
                tag.overrun = !boundary_announces_itself(tag.center_hz, next.center_hz);
                tag.center_hz = next.center_hz;
                segment += 1;
                continue;
            }
            let segment_end = segments.get(segment + 1).map_or(u64::MAX, |s| s.start);
            let take = (segment_end - position).min(rest.len() as u64) as usize;
            let expected = segment_end.min(file_samples).saturating_sub(position);
            if !packetizer.push(&rest[..take], expected, &mut tag) {
                return Ok(false);
            }
            rest = &rest[take..];
            position += take as u64;
        }
    }
}

/// Does a capture boundary from `from_hz` to `to_hz` show in the tags
/// alone? Only as a change between two *stated* frequencies. A boundary
/// that keeps the frequency, or that leaves or enters an unknown one,
/// is a break the consumer cannot see from the tag, so it is flagged.
fn boundary_announces_itself(from_hz: f64, to_hz: f64) -> bool {
    let stated = |f: f64| f != UNKNOWN_CENTER_HZ;
    stated(from_hz) && stated(to_hz) && from_hz != to_hz
}

/// The capture segments of a SigMF recording, normalised for
/// [`play_file`]: sorted by `core:sample_start`, starting at sample 0,
/// with strictly increasing starts. A capture without `core:frequency`
/// is tagged [`UNKNOWN_CENTER_HZ`], as is any stretch before the first
/// capture. Where two captures share a start the later one describes it.
fn sigmf_segments(meta: &SigmfMetadata) -> Vec<Segment> {
    let mut segments = vec![Segment {
        start: 0,
        center_hz: UNKNOWN_CENTER_HZ,
    }];
    for capture in meta.captures_by_start() {
        let seg = Segment {
            start: capture.sample_start,
            center_hz: capture.frequency.unwrap_or(UNKNOWN_CENTER_HZ),
        };
        match segments.last_mut() {
            Some(last) if last.start == seg.start => *last = seg,
            _ => segments.push(seg),
        }
    }
    segments
}

/// The on-disk format of a raw file, decided by its extension: `.bin` is
/// int16 scaled by 1/32768 (`ci16_le`); anything else is interleaved
/// `f32` (`cf32_le`).
fn raw_datatype(path: &Path) -> sigmf::DataType {
    if path.extension().is_some_and(|e| e == "bin") {
        sigmf::DataType::Ci16Le
    } else {
        sigmf::DataType::Cf32Le
    }
}

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
        let rate = packet_sample_rate(config.sample_rate_hz)
            .map_err(|e| SdrError::BadConfig(format!("RawIqFileSource: {e}")))?;
        let (tx, receiver) = channel::bounded::<IqPacket>(PACKET_QUEUE_CAPACITY);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop_flag.clone();
        let paths = self.paths.clone();
        let segments = [Segment {
            start: 0,
            center_hz: self.center_frequency_hz,
        }];

        let (pool_tx, pool_rx) = channel::bounded::<Vec<Complex32>>(PACKET_QUEUE_CAPACITY);

        let capture_thread = thread::spawn(move || {
            let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut packetizer = Packetizer {
                    acc: Vec::new(),
                    tx,
                    pool_tx,
                    pool_rx,
                };
                if let Err(e) = (move || -> Result<(), anyhow::Error> {
                    for path in paths {
                        if stop_for_thread.load(Ordering::SeqCst) {
                            break;
                        }
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
                        if !play_file(
                            &mut file,
                            &path,
                            raw_datatype(&path),
                            &segments,
                            rate,
                            &mut packetizer,
                            &stop_for_thread,
                        )? {
                            return Ok(());
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
/// A multi-capture recording is tagged per capture: packets are cut at
/// each capture's `core:sample_start` and carry that capture's
/// `core:frequency`, or [`UNKNOWN_CENTER_HZ`] where it states none. A
/// boundary the tags do not reveal — one that keeps the frequency, or
/// enters or leaves an unknown one — sets `overrun` on the first packet
/// after it.
pub struct SigmfFileSource {
    /// Each path is either a `.sigmf-meta`, a `.sigmf-data`, or a
    /// recording basename (which may contain dots) whose `.sigmf-meta` and
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
        let (tx, receiver) = channel::bounded::<IqPacket>(PACKET_QUEUE_CAPACITY);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop_flag.clone();
        let paths = self.paths.clone();

        let (pool_tx, pool_rx) = channel::bounded::<Vec<Complex32>>(PACKET_QUEUE_CAPACITY);

        let capture_thread = thread::spawn(move || {
            let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut packetizer = Packetizer {
                    acc: Vec::new(),
                    tx,
                    pool_tx,
                    pool_rx,
                };
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
                        let segments = sigmf_segments(&meta);
                        let unstated = segments
                            .iter()
                            .filter(|s| s.center_hz == UNKNOWN_CENTER_HZ)
                            .count();
                        if unstated > 0 {
                            warn!(
                                "sigmf: {}: {unstated} of {} capture segment(s) state no \
                                 core:frequency; their packets are tagged {UNKNOWN_CENTER_HZ} Hz \
                                 (unknown)",
                                meta_path.display(),
                                segments.len()
                            );
                        }
                        let center_hz = segments[0].center_hz;
                        let sample_rate = meta.sample_rate_hz();
                        let sample_rate_f32 = match packet_sample_rate(sample_rate) {
                            Ok(rate) => rate,
                            Err(e) => {
                                warn!("sigmf: {}: {e}; skipping file", meta_path.display());
                                continue;
                            }
                        };
                        info!(
                            "sigmf: playing {} ({}, {} MHz @ {:.3} MSPS, {} capture segment(s))",
                            data_path.display(),
                            meta.global.datatype,
                            center_hz / 1e6,
                            sample_rate / 1e6,
                            segments.len(),
                        );

                        let mut file = match File::open(&data_path) {
                            Ok(f) => f,
                            Err(e) => {
                                warn!("sigmf: failed to open {}: {e}", data_path.display());
                                continue;
                            }
                        };
                        if !play_file(
                            &mut file,
                            &data_path,
                            datatype,
                            &segments,
                            sample_rate_f32,
                            &mut packetizer,
                            &stop_for_thread,
                        )? {
                            return Ok(());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_f32_decodes_as_cf32_le() {
        // Encode two IQ pairs as little-endian f32, then decode.
        let samples = [Complex32::new(1.5, -2.5), Complex32::new(0.25, 0.5)];
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for s in samples {
            bytes.extend_from_slice(&s.re.to_le_bytes());
            bytes.extend_from_slice(&s.im.to_le_bytes());
        }
        let decoded = raw_datatype(Path::new("x.cf32")).decode(&bytes);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0].re - 1.5).abs() < 1e-6);
        assert!((decoded[0].im + 2.5).abs() < 1e-6);
        assert!((decoded[1].re - 0.25).abs() < 1e-6);
        assert!((decoded[1].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn raw_bin_decodes_as_int16_scaled() {
        // int16 32767 → ~1.0, -32768 → ~-1.0, scaled by 1/32768.
        let bytes = [
            0xFF, 0x7F, // re = 32767
            0x00, 0x80, // im = -32768
        ];
        let decoded = raw_datatype(Path::new("x.bin")).decode(&bytes);
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

    fn play_sigmf(dir: &Path, name: &str, total_samples: usize, captures: &str) -> Vec<IqPacket> {
        use std::time::Duration;
        let base = dir.join(name);
        let meta_path = PathBuf::from(format!("{}.sigmf-meta", base.display()));
        let data_path = PathBuf::from(format!("{}.sigmf-data", base.display()));
        std::fs::write(
            &meta_path,
            format!(
                r#"{{"global": {{"core:datatype": "cf32_le", "core:sample_rate": 1000000}},
                    "captures": {captures}}}"#
            ),
        )
        .unwrap();
        // The real part of every sample is its own index, so the stream is
        // its own record of where each packet started.
        let mut data_bytes = Vec::with_capacity(total_samples * 8);
        for i in 0..total_samples {
            data_bytes.extend_from_slice(&(i as f32).to_le_bytes());
            data_bytes.extend_from_slice(&0f32.to_le_bytes());
        }
        std::fs::write(&data_path, &data_bytes).unwrap();

        struct NoSignal;
        impl DwellAdvice for NoSignal {
            fn latest_signal_at(&self, _: u64) -> Option<std::time::Instant> {
                None
            }
        }
        let config = SourceConfig {
            sample_rate_hz: 0.0,
            channels_hz: vec![],
            dwell_min: Duration::ZERO,
            dwell_max: Duration::ZERO,
            dwell_extension: Duration::ZERO,
        };
        let handle = Box::new(SigmfFileSource {
            paths: vec![meta_path],
        })
        .start(config, Arc::new(NoSignal))
        .expect("start");
        let mut packets = Vec::new();
        while let Ok(pkt) = handle.receiver.recv_timeout(Duration::from_secs(2)) {
            packets.push(pkt);
        }
        packets
    }

    /// (first sample index, length, centre, overrun) of each packet.
    fn layout(packets: &[IqPacket]) -> Vec<(usize, usize, f64, bool)> {
        packets
            .iter()
            .map(|p| {
                (
                    p.samples[0].re as usize,
                    p.samples.len(),
                    p.center_frequency_hz,
                    p.overrun,
                )
            })
            .collect()
    }

    /// Pre-fix every packet carried the first capture's frequency, and
    /// the packet straddling the boundary mixed both captures' samples.
    #[test]
    fn each_capture_is_tagged_with_its_own_frequency_and_cut_at_its_start() {
        let dir = tempfile::tempdir().unwrap();
        let boundary = 200_000;
        let total = boundary + PACKET_SAMPLES + 5;
        let packets = play_sigmf(
            dir.path(),
            "two_captures",
            total,
            &format!(
                r#"[{{"core:sample_start": 0, "core:frequency": 851000000}},
                    {{"core:sample_start": {boundary}, "core:frequency": 852500000}}]"#
            ),
        );
        assert_eq!(
            layout(&packets),
            vec![
                (0, boundary, 851e6, false),
                (boundary, PACKET_SAMPLES, 852.5e6, false),
                (boundary + PACKET_SAMPLES, 5, 852.5e6, false),
            ]
        );
    }

    /// A new capture at the same frequency is still a break in the
    /// stream, and a boundary that falls exactly on a packet edge is not
    /// lost.
    #[test]
    fn a_same_frequency_capture_boundary_is_flagged_as_a_break() {
        let dir = tempfile::tempdir().unwrap();
        let packets = play_sigmf(
            dir.path(),
            "same_frequency",
            PACKET_SAMPLES + 300,
            &format!(
                r#"[{{"core:sample_start": 0, "core:frequency": 851000000}},
                    {{"core:sample_start": {PACKET_SAMPLES}, "core:frequency": 851000000}},
                    {{"core:sample_start": {}, "core:frequency": 851000000}}]"#,
                PACKET_SAMPLES + 100
            ),
        );
        assert_eq!(
            layout(&packets),
            vec![
                (0, PACKET_SAMPLES, 851e6, false),
                (PACKET_SAMPLES, 100, 851e6, true),
                (PACKET_SAMPLES + 100, 200, 851e6, true),
            ]
        );
    }

    /// A capture that states no frequency is tagged unknown, not with a
    /// neighbour's: the file's first frequency describes other samples.
    /// Both boundaries around it are flagged, because a consumer that
    /// ignores the unknown tag sees 851 MHz resume at 852.5 MHz's place
    /// with nothing in the tags to say the stream broke.
    #[test]
    fn a_capture_without_a_frequency_is_tagged_unknown_not_borrowed() {
        let dir = tempfile::tempdir().unwrap();
        let packets = play_sigmf(
            dir.path(),
            "unstated_frequency",
            PACKET_SAMPLES + 300,
            &format!(
                r#"[{{"core:sample_start": 0, "core:frequency": 851000000}},
                    {{"core:sample_start": {PACKET_SAMPLES}}},
                    {{"core:sample_start": {}, "core:frequency": 852500000}}]"#,
                PACKET_SAMPLES + 100
            ),
        );
        assert_eq!(
            layout(&packets),
            vec![
                (0, PACKET_SAMPLES, 851e6, false),
                (PACKET_SAMPLES, 100, UNKNOWN_CENTER_HZ, true),
                (PACKET_SAMPLES + 100, 200, 852.5e6, true),
            ]
        );
    }

    /// Captures listed out of order, a leading stretch no capture
    /// covers, and two captures sharing a start all normalise to one
    /// segment per distinct start — and the recording's reported centre
    /// frequency is the earliest capture's, in the same order playback
    /// uses, not the first listed.
    #[test]
    fn capture_segments_are_normalised() {
        let meta: SigmfMetadata = serde_json::from_str(
            r#"{"global": {"core:datatype": "cf32_le", "core:sample_rate": 1},
                "captures": [
                    {"core:sample_start": 50, "core:frequency": 3.0},
                    {"core:sample_start": 10, "core:frequency": 1.0},
                    {"core:sample_start": 50, "core:frequency": 4.0},
                    {"core:sample_start": 90}
                ]}"#,
        )
        .unwrap();
        let seg = |start, center_hz| Segment { start, center_hz };
        let unknown = UNKNOWN_CENTER_HZ;
        assert_eq!(
            sigmf_segments(&meta),
            vec![
                seg(0, unknown),
                seg(10, 1.0),
                seg(50, 4.0),
                seg(90, unknown)
            ]
        );
        assert_eq!(meta.center_frequency_hz(), Some(1.0));
    }

    /// Packets are the same whether a packet fills in one read or across
    /// many (1 MiB reads hold 131 072 cf32 samples; a packet is eight).
    #[test]
    fn packets_are_contiguous_and_complete_across_reads() {
        let dir = tempfile::tempdir().unwrap();
        let total = 2 * PACKET_SAMPLES + 131_072 / 2 + 3;
        let packets = play_sigmf(
            dir.path(),
            "contiguous",
            total,
            r#"[{"core:sample_start": 0, "core:frequency": 1.0}]"#,
        );
        let mut expected = 0usize;
        for p in &packets {
            for s in p.samples.iter() {
                assert_eq!(s.re as usize, expected);
                expected += 1;
            }
        }
        assert_eq!(expected, total);
        assert_eq!(
            packets.iter().map(|p| p.samples.len()).collect::<Vec<_>>(),
            vec![PACKET_SAMPLES, PACKET_SAMPLES, 131_072 / 2 + 3]
        );
    }
}

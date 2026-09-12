use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::channel::RecvTimeoutError;
use num_complex::Complex32;
use orecchiette_sdr_file_rs::sigmf::{
    DataType, SigmfCapture, SigmfMetadata, SigmfWriter, SigmfWriterMeta, looks_like_sigmf,
    resolve_pair,
};
use orecchiette_sdr_file_rs::{RawIqFileSource, SigmfFileSource};
use orecchiette_sdr_source_rs::{
    DwellAdvice, IqPacket, SdrError, SdrHandle, SdrSource, SourceConfig,
};

struct Advice;
impl DwellAdvice for Advice {
    fn latest_signal_at(&self, _: u64) -> Option<Instant> {
        None
    }
}
fn config(rate: f64) -> SourceConfig {
    SourceConfig {
        sample_rate_hz: rate,
        channels_hz: vec![],
        dwell_min: Duration::ZERO,
        dwell_max: Duration::ZERO,
        dwell_extension: Duration::ZERO,
    }
}
fn writer_meta() -> SigmfWriterMeta {
    SigmfWriterMeta {
        sample_rate_hz: 1e6,
        hardware: None,
        description: None,
        recorder: None,
        captures: vec![],
        annotations: vec![],
    }
}
fn drain(handle: SdrHandle) -> Vec<IqPacket> {
    let mut packets = vec![];
    let end = loop {
        match handle.receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(packet) => packets.push(packet),
            Err(e) => break e,
        }
    };
    (handle.stop)();
    drop(handle.receiver);
    (handle.wait)();
    assert_eq!(
        end,
        RecvTimeoutError::Disconnected,
        "playback did not finish"
    );
    packets
}
fn write_fixture(base: &Path, rate: f64, channels: Option<serde_json::Value>) {
    let mut global = serde_json::json!({"core:datatype": "cf32_le", "core:sample_rate": rate});
    if let Some(channels) = channels {
        global["core:num_channels"] = channels;
    }
    std::fs::write(
        base.with_extension("sigmf-meta"),
        serde_json::to_vec(&serde_json::json!({
            "global": global, "captures": [], "annotations": []
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(base.with_extension("sigmf-data"), [0u8; 8]).unwrap();
}

#[test]
fn distinct_dotted_recordings_do_not_overwrite_each_other() {
    let dir = tempfile::tempdir().unwrap();
    let names = [
        "capture.001",
        "capture.002",
        "capture.003.sigmf-meta",
        "capture.004.sigmf-data",
    ];
    for (i, name) in names.iter().enumerate() {
        let mut writer = SigmfWriter::create(&dir.path().join(name), DataType::Cf32Le).unwrap();
        writer
            .write_samples(&[Complex32::new(i as f32, -(i as f32))])
            .unwrap();
        writer.finalize(writer_meta()).unwrap();
    }
    for i in 0..names.len() {
        let base = dir.path().join(format!("capture.{:03}", i + 1));
        let meta = dir.path().join(format!("capture.{:03}.sigmf-meta", i + 1));
        let data = dir.path().join(format!("capture.{:03}.sigmf-data", i + 1));
        for input in [&base, &meta, &data] {
            assert!(looks_like_sigmf(input));
            assert_eq!(resolve_pair(input).unwrap(), (meta.clone(), data.clone()));
        }
        assert_eq!(
            DataType::Cf32Le.decode(&std::fs::read(data).unwrap()),
            vec![Complex32::new(i as f32, -(i as f32))]
        );
    }
}

#[test]
fn writer_preserves_signed_and_fractional_frequencies() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("frequency");
    let frequencies = [-1250.25, 0.0, 915_000_000.125];
    let mut meta = writer_meta();
    meta.captures = frequencies
        .iter()
        .enumerate()
        .map(|(i, &frequency)| SigmfCapture {
            sample_start: i as u64,
            frequency_hz: Some(frequency),
            ..Default::default()
        })
        .collect();
    let mut writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
    writer
        .write_samples(&[Complex32::new(0.0, 0.0); 3])
        .unwrap();
    writer.finalize(meta).unwrap();
    let loaded = SigmfMetadata::load(&base.with_extension("sigmf-meta")).unwrap();
    assert_eq!(
        loaded
            .captures
            .iter()
            .map(|c| c.frequency.unwrap())
            .collect::<Vec<_>>(),
        frequencies
    );
}

#[test]
fn raw_source_rejects_rates_that_cannot_be_used_in_packets() {
    for rate in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.0,
        1e300,
        1e-300,
    ] {
        let result = Box::new(RawIqFileSource {
            paths: vec!["unused.cf32".into()],
            center_frequency_hz: 0.0,
        })
        .start(config(rate), Arc::new(Advice));
        assert!(
            matches!(result, Err(SdrError::BadConfig(_))),
            "accepted {rate}"
        );
    }
}

#[test]
fn sigmf_source_skips_invalid_rates_and_plays_the_next_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = vec![];
    for (i, rate) in [0.0, -1.0, 1e300, 1e-300, 1e6].iter().enumerate() {
        let base = dir.path().join(format!("rate{i}"));
        write_fixture(&base, *rate, None);
        paths.push(base);
    }
    let packets = drain(
        Box::new(SigmfFileSource { paths })
            .start(config(0.0), Arc::new(Advice))
            .unwrap(),
    );
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].sample_rate_hz, 1e6);
}

#[test]
fn valid_extreme_packet_rates_still_play() {
    let dir = tempfile::tempdir().unwrap();
    for rate in [f32::from_bits(1), 1e6, f32::MAX] {
        let base = dir.path().join("valid");
        write_fixture(&base, rate as f64, None);
        let raw = Box::new(RawIqFileSource {
            paths: vec![base.with_extension("sigmf-data")],
            center_frequency_hz: 0.0,
        });
        let sigmf = Box::new(SigmfFileSource { paths: vec![base] });
        for source in [raw as Box<dyn SdrSource>, sigmf] {
            let packets = drain(source.start(config(rate as f64), Arc::new(Advice)).unwrap());
            assert_eq!(packets.len(), 1);
            assert_eq!(packets[0].sample_rate_hz, rate);
        }
    }
}

#[test]
fn reader_rejects_unsupported_channel_counts_before_playback() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = vec![];
    for (i, channels) in [
        serde_json::json!(0),
        serde_json::json!(2),
        serde_json::json!(-1),
        serde_json::json!(1.5),
        serde_json::json!(null),
        serde_json::json!("1"),
    ]
    .into_iter()
    .enumerate()
    {
        let base = dir.path().join(format!("channels{i}"));
        write_fixture(&base, 1e6, Some(channels));
        assert!(SigmfMetadata::load(&base.with_extension("sigmf-meta")).is_err());
        paths.push(base);
    }
    for (i, channels) in [None, Some(serde_json::json!(1))].into_iter().enumerate() {
        let base = dir.path().join(format!("single{i}"));
        write_fixture(&base, 1e6, channels);
        assert!(SigmfMetadata::load(&base.with_extension("sigmf-meta")).is_ok());
        paths.push(base);
    }
    let packets = drain(
        Box::new(SigmfFileSource { paths })
            .start(config(0.0), Arc::new(Advice))
            .unwrap(),
    );
    assert_eq!(packets.len(), 2, "unsupported files must emit no samples");
}

#[test]
fn failed_numeric_validation_does_not_overwrite_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("invalid");
    let meta_path = base.with_extension("sigmf-meta");
    let mut invalid = vec![];
    for rate in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.0,
        1e300,
        1e-300,
    ] {
        let mut meta = writer_meta();
        meta.sample_rate_hz = rate;
        invalid.push(meta);
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for capture in [
            SigmfCapture {
                frequency_hz: Some(value),
                ..Default::default()
            },
            SigmfCapture {
                geolocation: Some([value, 0.0]),
                ..Default::default()
            },
            SigmfCapture {
                geolocation: Some([0.0, value]),
                ..Default::default()
            },
        ] {
            let mut meta = writer_meta();
            meta.captures.push(capture);
            invalid.push(meta);
        }
    }
    for meta in invalid {
        std::fs::write(&meta_path, "previous metadata").unwrap();
        let writer = SigmfWriter::create(&base, DataType::Cf32Le).unwrap();
        assert!(writer.finalize(meta).is_err());
        assert_eq!(
            std::fs::read_to_string(&meta_path).unwrap(),
            "previous metadata"
        );
    }
}

// A separate test executable isolates allocator observations from other tests.
use orecchiette_sdr_file_rs::{RawIqFileSource, SigmfFileSource};
use orecchiette_sdr_source_rs::{DwellAdvice, SdrSource, SourceConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct CountingAllocator;
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= 8 * 1024 * 1024 {
            LARGE_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: forward the caller's valid allocation layout unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: allocations originate from System with the same layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
struct Advice;
impl DwellAdvice for Advice {
    fn latest_signal_at(&self, _: u64) -> Option<Instant> {
        None
    }
}
#[test]
fn tiny_playback_does_not_allocate_full_packet_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("tiny.cf32");
    std::fs::write(&raw, [0u8; 8]).unwrap();
    let meta = dir.path().join("tiny.sigmf-meta");
    std::fs::write(
        &meta,
        r#"{"global":{"core:datatype":"cf32_le","core:sample_rate":1000000}}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("tiny.sigmf-data"), [0u8; 8]).unwrap();
    let sources: Vec<Box<dyn SdrSource>> = vec![
        Box::new(RawIqFileSource {
            paths: vec![raw],
            center_frequency_hz: 0.0,
        }),
        Box::new(SigmfFileSource { paths: vec![meta] }),
    ];
    for source in sources {
        let before = LARGE_ALLOCATIONS.load(Ordering::SeqCst);
        let handle = source
            .start(
                SourceConfig {
                    sample_rate_hz: 1e6,
                    channels_hz: vec![],
                    dwell_min: Duration::ZERO,
                    dwell_max: Duration::ZERO,
                    dwell_extension: Duration::ZERO,
                },
                Arc::new(Advice),
            )
            .unwrap();
        let capacity = handle.receiver.capacity().unwrap();
        let packet = handle.receiver.recv_timeout(Duration::from_secs(5));
        (handle.stop)();
        drop(handle.receiver);
        (handle.wait)();
        assert_eq!(packet.unwrap().samples.len(), 1);
        assert!(
            capacity <= 8,
            "queue can retain more than 64 MiB of full packets"
        );
        assert_eq!(
            LARGE_ALLOCATIONS.load(Ordering::SeqCst),
            before,
            "one-sample playback must not reserve full 8 MiB packet buffers"
        );
    }
}

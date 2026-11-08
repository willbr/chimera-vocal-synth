//! The Rust side of `n64tts`, the C library that would run on the console.

pub const ORDER: usize = 16;
pub const OUT_RATE: u32 = 16000;
pub const FRAME_BYTES: usize = ORDER + 2;
pub const PULSE_LEN: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub samples: u32,
    /// Hz in Q4, at the middle of the sound
    pub f0: u32,
    pub phone: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub size: i32,
    pub breath: i32,
    pub bright: i32,
    pub rough: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub samples: u32,
    pub filter_updates: u32,
    pub joins: u32,
    pub borrowed_joins: u32,
}

#[repr(C)]
struct RawVoice {
    cos_table: *const u8,
    pulse: *const u8,
    index: *const u8,
    frames: *const u8,
    joins: u32,
    frame_count: u32,
}

unsafe extern "C" {
    fn n64tts_voice_init(v: *mut RawVoice, data: *const u8, length: u32) -> i32;
    fn n64tts_output_len(events: *const Event, count: u32) -> u32;
    fn n64tts_render(v: *const RawVoice, events: *const Event, count: u32, params: *const Params, out: *mut i16, stats: *mut Stats);
}

/// A voice file in memory, and the library's view of it.
pub struct Voice {
    _data: Vec<u8>,
    raw: RawVoice,
}

impl Voice {
    pub fn from_bytes(data: Vec<u8>) -> Option<Voice> {
        let mut raw = RawVoice { cos_table: std::ptr::null(), pulse: std::ptr::null(), index: std::ptr::null(), frames: std::ptr::null(), joins: 0, frame_count: 0 };
        // SAFETY: `data` is a live buffer of the length given, and init only reads it. `raw` then
        // points into the Vec's heap allocation, which stays put when the Vec moves into `Voice`.
        let status = unsafe { n64tts_voice_init(&mut raw, data.as_ptr(), data.len() as u32) };
        (status == 0).then_some(Voice { _data: data, raw })
    }

    pub fn joins(&self) -> u32 {
        self.raw.joins
    }
}

pub fn output_len(events: &[Event]) -> usize {
    // SAFETY: the pointer and count describe `events`.
    unsafe { n64tts_output_len(events.as_ptr(), events.len() as u32) as usize }
}

/// `out` must be at least `output_len(events)` long.
pub fn render(voice: &Voice, events: &[Event], params: &Params, out: &mut [i16]) -> Stats {
    assert!(out.len() >= output_len(events));
    let mut stats = Stats::default();
    // SAFETY: every pointer is valid for the call, and `out` was checked to be long enough.
    unsafe { n64tts_render(&voice.raw, events.as_ptr(), events.len() as u32, params, out.as_mut_ptr(), &mut stats) };
    stats
}

/// The working memory one render uses, in bytes: filter state and the list of sounds.
pub fn ram_bytes(events: usize) -> usize {
    ORDER * 8 * 2 + ORDER * 4 * 2 + 128 + events * std::mem::size_of::<Event>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sounds, with joins 0→1 and 1→0 only, four frames each.
    fn tiny_voice() -> Voice {
        let mut b = b"CVOX\x02\x01a\x01b".to_vec();
        for k in 0..=256 {
            let c = (std::f64::consts::PI * k as f64 / 256.0).cos() * 32767.0;
            b.extend_from_slice(&(c.round() as i16).to_le_bytes());
        }
        for k in 0..PULSE_LEN as i32 {
            b.extend_from_slice(&(if k == 64 { 20000i16 } else { -157 }).to_le_bytes());
        }
        b.extend_from_slice(&2u16.to_le_bytes());
        for (i, (x, y)) in [(0u8, 1u8), (1, 0)].into_iter().enumerate() {
            b.extend_from_slice(&[x, y]);
            b.extend_from_slice(&(i as u32 * 4).to_le_bytes());
            b.extend_from_slice(&4u16.to_le_bytes());
            b.extend_from_slice(&2u16.to_le_bytes());
        }
        for _ in 0..8 {
            b.extend((0..ORDER).map(|k| (k * 15 + 8) as u8));
            b.extend_from_slice(&[120, 15]);
        }
        Voice::from_bytes(b).unwrap()
    }

    fn say(voice: &Voice, phones: &[u8]) -> (Vec<i16>, Stats) {
        let events: Vec<Event> = phones.iter().map(|&phone| Event { samples: 1600, f0: 120 * 16, phone }).collect();
        let params = Params { size: 4096, breath: 200, bright: 0, rough: 400 };
        let mut out = vec![0i16; output_len(&events)];
        let stats = render(voice, &events, &params, &mut out);
        (out, stats)
    }

    #[test]
    fn the_same_input_gives_the_same_samples() {
        let voice = tiny_voice();
        let (a, stats) = say(&voice, &[0, 1, 0]);
        assert_eq!(a, say(&voice, &[0, 1, 0]).0);
        assert_eq!(a.len(), 3200);
        assert_eq!(stats.borrowed_joins, 0);
        assert!(a.iter().any(|&s| s != 0), "silent output");
    }

    #[test]
    fn a_join_never_recorded_borrows_from_others() {
        let (a, stats) = say(&tiny_voice(), &[0, 0]);
        assert_eq!(stats.borrowed_joins, 1);
        assert!(a.iter().any(|&s| s != 0), "silent output");
    }

    #[test]
    fn the_library_has_no_floating_point() {
        let source = include_str!("../n64tts/n64tts.c");
        for bad in ["float", "double"] {
            assert!(!source.split("See n64tts.h. Integer only: a test in the host tool forbids floating-point type names here.").last().unwrap().contains(bad), "{bad} appears in the console code");
        }
    }
}

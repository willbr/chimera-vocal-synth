//! Text and a recording in; the engine's render, the recording's phoneme timing, and a comparison
//! out. Whisper only supplies word times. Phoneme times come from warping the recording onto the
//! engine's render of the same text, whose phoneme times the plan already knows.

use chimera::analysis::fft::{Fft, hann, next_pow2};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::english::{EnglishG2P, Phrase};
use chimera::phonetics::planner::{PhraseInput, PlanOptions, expand, plan_phrases};
use chimera::phonetics::{Plan, PlannedSegment};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::io::Read;
use std::path::Path;
use std::process::Command;

pub const FS: f64 = 48000.0;
const LOW_FS: f64 = 11025.0;
const HOP_S: f64 = 0.01;
const N_MEL: usize = 40;
const N_CEP: usize = 13;
/// How far (s) the warp may stray from Whisper's word times.
const ANCHORED_BAND_S: f64 = 0.25;
const MAX_RECORDING_S: f64 = 60.0;

pub fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

// ---- audio files

pub fn resample(x: &[f32], from: f64, to: f64) -> Vec<f32> {
    if (from - to).abs() < 1e-6 {
        return x.to_vec();
    }
    let ratio = to / from;
    let cutoff = ratio.min(1.0) * 0.98;
    let half = (16.0 / cutoff).ceil() as isize;
    let n = (x.len() as f64 * ratio).floor() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / ratio;
            let c = t.floor() as isize;
            let mut s = 0.0;
            for k in c - half + 1..=c + half {
                if k < 0 || k as usize >= x.len() {
                    continue;
                }
                let d = t - k as f64;
                let w = 0.5 + 0.5 * (PI * d / half as f64).cos();
                let a = PI * d * cutoff;
                let sinc = if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                s += x[k as usize] as f64 * sinc * cutoff * w;
            }
            s as f32
        })
        .collect()
}

/// Any PCM or float WAV as mono at `FS`.
pub fn read_wav(path: &str) -> Result<Vec<f32>, String> {
    let mut r = hound::WavReader::open(path).map_err(|e| format!("{path}: {e}"))?;
    let spec = r.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let full = (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / full).collect()
        }
    };
    let ch = spec.channels.max(1) as usize;
    let mono: Vec<f32> = interleaved.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    Ok(resample(&mono, spec.sample_rate as f64, FS))
}

pub fn write_wav(path: &str, x: &[f32]) -> Result<(), String> {
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let spec = hound::WavSpec { channels: 1, sample_rate: FS as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| format!("{path}: {e}"))?;
    for &s in x {
        let v = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        w.write_sample((v * 32767.0).round() as i16).map_err(|e| format!("{path}: {e}"))?;
    }
    w.finalize().map_err(|e| format!("{path}: {e}"))
}

// ---- engine

pub fn load_g2p() -> Result<EnglishG2P, String> {
    let path = repo_path("public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).map_err(|e| format!("{path}: {e}"))?;
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    Ok(g)
}

pub(crate) fn inputs(phrases: &[Phrase]) -> Vec<PhraseInput> {
    phrases.iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect()
}

pub(crate) fn plan(inputs: &[PhraseInput], body: &str, f0: f64) -> Plan {
    let mix: BodyMix = body.into();
    plan_phrases(inputs, &PlanOptions { body: mix, t0: 0.05, f0: Some(f0), ..Default::default() })
}

fn render(plan: &Plan, body: &str, f0: f64) -> Vec<f32> {
    let mix: BodyMix = body.into();
    let mut voice = HumanVoice::new(FS, mix.body(), 2);
    voice.set_source(&SourcePatch::f0(f0));
    voice.schedule(&plan.gestures);
    let mut out = vec![0f32; (FS * (plan.end + 0.3)).round() as usize];
    for block in out.chunks_mut(128) {
        voice.render(block);
    }
    out
}

/// One input segment's slice of `plan.segments`, and which word it belongs to.
#[derive(Clone, Copy)]
pub(crate) struct SegSpan {
    pub(crate) phrase: usize,
    pub(crate) seg: usize,
    pub(crate) first: usize,
    pub(crate) count: usize,
    pub(crate) word: usize,
}

pub(crate) fn seg_spans(phrases: &[Phrase], planned: &[PlannedSegment]) -> Result<Vec<SegSpan>, String> {
    let mut out = Vec::new();
    let (mut first, mut word) = (0, 0);
    for (pi, p) in phrases.iter().enumerate() {
        for (si, s) in p.segments.iter().enumerate() {
            let count = expand(&s.ipa).map_or(1, |parts| parts.len());
            out.push(SegSpan { phrase: pi, seg: si, first, count, word });
            first += count;
            if s.word_end == Some(true) {
                word += 1;
            }
        }
    }
    if first != planned.len() {
        return Err(format!("planner produced {} segments, expected {first}", planned.len()));
    }
    Ok(out)
}

// ---- Whisper

#[derive(Clone, Debug, Deserialize)]
struct WhisperWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Deserialize)]
struct WhisperFile {
    words: Vec<WhisperWord>,
}

fn run_whisper(wav: &str, out_json: &str) -> Result<Vec<WhisperWord>, String> {
    let script = format!("{}/whisper_words.py", env!("CARGO_MANIFEST_DIR"));
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--no-project", "--quiet", "--python", "3.12", "--with", "faster-whisper==1.2.1"]).arg(&script).arg(wav).arg(out_json);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = cmd.output().map_err(|e| format!("couldn't start uv (needed for Whisper): {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = err.lines().rev().take(8).collect();
        return Err(format!("Whisper failed:\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n")));
    }
    let text = std::fs::read_to_string(out_json).map_err(|e| format!("{out_json}: {e}"))?;
    let file: WhisperFile = serde_json::from_str(&text).map_err(|e| format!("{out_json}: {e}"))?;
    Ok(file.words)
}

/// Is the model already in the cache `whisper_words.py` downloads it to?
fn whisper_model_cached() -> bool {
    let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) else { return false };
    let Ok(entries) = std::fs::read_dir(format!("{home}/.cache/chimera-asr/models")) else { return false };
    entries.flatten().any(|e| e.file_name().to_string_lossy().contains("small.en"))
}

fn norm_word(w: &str) -> String {
    w.to_lowercase().replace('’', "'").chars().filter(|c| c.is_ascii_lowercase() || *c == '\'').collect::<String>().trim_matches('\'').to_string()
}

/// Longest common subsequence of said and heard words: (said index, heard index) pairs.
fn match_words(said: &[String], heard: &[WhisperWord]) -> Vec<(usize, usize)> {
    let said: Vec<String> = said.iter().map(|w| norm_word(w)).collect();
    let heard: Vec<String> = heard.iter().map(|w| norm_word(&w.word)).collect();
    let (n, m) = (said.len(), heard.len());
    let mut t = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[i][j] = if said[i] == heard[j] { t[i + 1][j + 1] + 1 } else { t[i + 1][j].max(t[i][j + 1]) };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n && j < m {
        if said[i] == heard[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if t[i + 1][j] >= t[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

// ---- alignment

/// Log-mel cepstra, 10 ms hop, normalized per utterance so a microphone and a dry render compare.
fn features(x: &[f32]) -> Vec<Vec<f64>> {
    let win_len = (0.025 * FS) as usize;
    let n = next_pow2(win_len as f64);
    let fft = Fft::new(n);
    let win = hann(win_len);
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let (m0, m1) = (hz_to_mel(100.0), hz_to_mel(7500.0));
    let edge = |k: usize| 700.0 * (10f64.powf((m0 + (m1 - m0) * k as f64 / (N_MEL + 1) as f64) / 2595.0) - 1.0);
    let bin = FS / n as f64;
    let filters: Vec<Vec<(usize, f64)>> = (0..N_MEL)
        .map(|k| {
            let (lo, mid, hi) = (edge(k), edge(k + 1), edge(k + 2));
            (0..=n / 2)
                .filter_map(|i| {
                    let f = i as f64 * bin;
                    let w = if f > lo && f <= mid {
                        (f - lo) / (mid - lo)
                    } else if f > mid && f < hi {
                        (hi - f) / (hi - mid)
                    } else {
                        0.0
                    };
                    (w > 0.0).then_some((i, w))
                })
                .collect()
        })
        .collect();
    let hop = (HOP_S * FS) as usize;
    let mut frames: Vec<Vec<f64>> = Vec::new();
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    let mut start = 0;
    while start + win_len <= x.len() {
        re.fill(0.0);
        im.fill(0.0);
        for i in 0..win_len {
            re[i] = x[start + i] as f64 * win[i];
        }
        fft.transform(&mut re, &mut im);
        let mel: Vec<f64> = filters.iter().map(|f| (f.iter().map(|&(i, w)| w * (re[i] * re[i] + im[i] * im[i])).sum::<f64>() + 1e-10).ln()).collect();
        let cep: Vec<f64> = (0..N_CEP).map(|c| mel.iter().enumerate().map(|(j, &v)| v * (PI * c as f64 * (j as f64 + 0.5) / N_MEL as f64).cos()).sum::<f64>()).collect();
        frames.push(cep);
        start += hop;
    }
    if frames.is_empty() {
        return frames;
    }
    for c in 0..N_CEP {
        let mean = frames.iter().map(|f| f[c]).sum::<f64>() / frames.len() as f64;
        let var = frames.iter().map(|f| (f[c] - mean).powi(2)).sum::<f64>() / frames.len() as f64;
        let sd = var.sqrt().max(1e-6);
        for f in frames.iter_mut() {
            f[c] = (f[c] - mean) / sd;
        }
    }
    // how fast each coefficient is changing: this is what marks a phoneme boundary
    let last = frames.len() - 1;
    let deltas: Vec<Vec<f64>> = (0..frames.len())
        .map(|t| {
            let (a, b) = (&frames[t.saturating_sub(2)], &frames[(t + 2).min(last)]);
            (0..N_CEP).map(|c| b[c] - a[c]).collect()
        })
        .collect();
    for (f, d) in frames.iter_mut().zip(deltas) {
        f.extend(d);
    }
    frames
}

/// For each engine frame, the recording frame it lands on, inside a band around `guide`.
fn warp(eng: &[Vec<f64>], mine: &[Vec<f64>], guide: &dyn Fn(f64) -> f64, band: usize) -> Vec<f64> {
    let (e, m) = (eng.len(), mine.len());
    let mut lo: Vec<usize> = Vec::with_capacity(e);
    let mut hi: Vec<usize> = Vec::with_capacity(e);
    for i in 0..e {
        let c = guide(i as f64 * HOP_S) / HOP_S;
        let c = c.clamp(0.0, (m - 1) as f64) as usize;
        lo.push(c.saturating_sub(band));
        hi.push((c + band).min(m - 1));
    }
    // every row must be reachable from the one before
    for i in (0..e - 1).rev() {
        hi[i] = hi[i].max(lo[i + 1]);
    }
    let w = e * m;
    let mut d = vec![f32::INFINITY; w];
    let mut back = vec![0u8; w];
    let at = |i: usize, j: usize| i * m + j;
    let cost = |i: usize, j: usize| eng[i].iter().zip(&mine[j]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() as f32;
    for i in 0..e {
        for j in lo[i]..=hi[i] {
            let c = cost(i, j);
            if i == 0 {
                // free start inside the band: a recording usually opens with some silence
                d[at(i, j)] = c;
                continue;
            }
            // Slopes are limited to between 1/2 and 2, so no phoneme can collapse to nothing or
            // swallow its neighbours, and every step's cost is weighted by the frames it covers
            // (unweighted, the diagonal is cheapest and the path just returns the engine's timing).
            let diag = if j >= 1 { d[at(i - 1, j - 1)] + 2.0 * c } else { f32::INFINITY };
            let fast = if i >= 2 && j >= 1 { d[at(i - 2, j - 1)] + 2.0 * cost(i - 1, j) + c } else { f32::INFINITY };
            let slow = if j >= 2 { d[at(i - 1, j - 2)] + 2.0 * cost(i, j - 1) + c } else { f32::INFINITY };
            let (best, step) = if diag <= fast && diag <= slow {
                (diag, 0)
            } else if fast <= slow {
                (fast, 1)
            } else {
                (slow, 2)
            };
            d[at(i, j)] = best;
            back[at(i, j)] = step;
        }
    }
    let mut j = (lo[e - 1]..=hi[e - 1]).min_by(|&a, &b| d[at(e - 1, a)].total_cmp(&d[at(e - 1, b)])).unwrap_or(m - 1);
    if !d[at(e - 1, j)].is_finite() {
        eprintln!("no warp fits the slope limits; falling back to the word timings alone");
        return (0..e).map(|i| guide(i as f64 * HOP_S) / HOP_S).collect();
    }
    let mut i = e - 1;
    let (mut sum, mut count) = (vec![0.0f64; e], vec![0u32; e]);
    let mut add = |i: usize, j: usize| {
        sum[i] += j as f64;
        count[i] += 1;
    };
    loop {
        add(i, j);
        if i == 0 {
            break;
        }
        match back[at(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => {
                add(i - 1, j);
                i -= 2;
                j -= 1;
            }
            _ => {
                add(i, j - 1);
                i -= 1;
                j -= 2;
            }
        }
    }
    (0..e).map(|i| sum[i] / count[i].max(1) as f64).collect()
}

fn interp(points: &[(f64, f64)], t: f64) -> f64 {
    if t <= points[0].0 {
        return points[0].1;
    }
    for w in points.windows(2) {
        if t <= w[1].0 {
            let span = (w[1].0 - w[0].0).max(1e-9);
            return w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / span;
        }
    }
    points[points.len() - 1].1
}

/// The warp from engine frames to recording frames, guided by (engine time, recording time) points.
fn align(eng: &[f32], mine: &[f32], mut points: Vec<(f64, f64)>) -> Result<Vec<f64>, String> {
    let (mine_dur, eng_dur) = (mine.len() as f64 / FS, eng.len() as f64 / FS);
    let anchored = !points.is_empty();
    if anchored {
        let (e, m) = points[0];
        points.insert(0, (0.0, (m - e).max(0.0)));
        let (e, m) = points[points.len() - 1];
        points.push((eng_dur.max(e + 0.01), (m + (eng_dur - e)).clamp(m, mine_dur.max(m))));
    } else {
        points = vec![(0.0, 0.0), (eng_dur, mine_dur)];
    }
    let (fe, fm) = (features(eng), features(mine));
    if fe.is_empty() || fm.is_empty() {
        return Err("recording is too short to analyze".to_string());
    }
    let band = if anchored { (ANCHORED_BAND_S / HOP_S) as usize } else { 25 + fm.len() / 5 };
    Ok(warp(&fe, &fm, &|t| interp(&points, t), band))
}

fn map_time(path: &[f64], t: f64) -> f64 {
    let f = (t / HOP_S).clamp(0.0, (path.len() - 1) as f64);
    let (i, frac) = (f.floor() as usize, f.fract());
    let next = path[(i + 1).min(path.len() - 1)];
    (path[i] + (next - path[i]) * frac) * HOP_S
}

/// Can the warp recover timing it should be able to? The "speaker" is the engine in the other
/// body at another pitch with every segment's duration scrambled, so the true boundaries are known.
pub fn self_test() -> Result<(), String> {
    let g2p = load_g2p()?;
    // finding the vowel in a word: the engine says each word, so where the vowel is, is known
    println!("words: vowel found by loudness vs measured at the vowel's known time (F1 F2 F3)");
    for (_, ipa, word, onset) in WORDS {
        let phrases = crate::british::southern_british(&g2p, &g2p.to_phrases(word), "male", 112.0, false)?;
        let p = plan(&inputs(&phrases), "male", 112.0);
        let audio = render(&p, "male", 112.0);
        let low: Vec<f64> = resample(&audio, FS, LOW_FS).iter().map(|&v| v as f64).collect();
        let v = p.segments.iter().find(|s| s.kind == "vowel").ok_or("no vowel planned")?;
        let known = formants(&low, v.t0, v.t1);
        match (measure_word(&audio, onset), known) {
            (Some(h), Some(k)) => println!("  {word:6} {ipa:2} found {:5.0} {:5.0} {:5.0} | known {:5.0} {:5.0} {:5.0} | off by {:+5.0} {:+5.0} {:+5.0} Hz", h.formants[0], h.formants[1], h.formants[2], k[0], k[1], k[2], h.formants[0] - k[0], h.formants[1] - k[1], h.formants[2] - k[2]),
            (h, k) => println!("  {word:6} {ipa:2} found {:?} known {k:?}", h.map(|h| h.formants)),
        }
    }
    for text in ["She had your dark suit in greasy wash water all year.", "Don't ask me to carry an oily rag like that."] {
        let phrases = g2p.to_phrases(text);
        let base = inputs(&phrases);
        let eng_plan = plan(&base, "male", 112.0);
        let eng = render(&eng_plan, "male", 112.0);
        let spans = seg_spans(&phrases, &eng_plan.segments)?;
        let mut scrambled = base.clone();
        let mut seed = 12345u32;
        for span in &spans {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = 0.6 + (seed >> 16) as f64 / 65536.0;
            let a = eng_plan.segments[span.first].t0;
            let b = eng_plan.segments[span.first + span.count - 1].t1;
            scrambled[span.phrase].segments[span.seg].dur = Some((b - a) * scale);
        }
        let truth_plan = plan(&scrambled, "female", 190.0);
        let truth = render(&truth_plan, "female", 190.0);
        if truth_plan.segments.len() != eng_plan.segments.len() {
            return Err("scrambled plan changed the segment count".to_string());
        }
        // word anchors like Whisper's: the true word times, off by up to 40 ms
        let mut points: Vec<(f64, f64)> = Vec::new();
        let words = spans.iter().map(|s| s.word).max().unwrap_or(0) + 1;
        for w in 0..words {
            let mut it = spans.iter().filter(|s| s.word == w);
            let Some(first) = it.next() else { continue };
            let last = it.last().unwrap_or(first);
            let (i, j) = (first.first, last.first + last.count - 1);
            let mut jitter = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f64 / 65536.0 - 0.5) * 0.08
            };
            let (m0, m1) = (truth_plan.segments[i].t0 + jitter(), truth_plan.segments[j].t1 + jitter());
            let prev = points.last().map_or(f64::NEG_INFINITY, |p| p.1);
            if m0 >= prev && m1 > m0 {
                points.push((eng_plan.segments[i].t0, m0));
                points.push((eng_plan.segments[j].t1, m1));
            }
        }
        let span_scale = truth_plan.end / eng_plan.end;
        let stats = |name: &str, map: &dyn Fn(f64) -> f64| {
            let mut edge: Vec<f64> = Vec::new();
            let mut dur: Vec<f64> = Vec::new();
            for (e, t) in eng_plan.segments.iter().zip(&truth_plan.segments) {
                let (a, b) = (map(e.t0), map(e.t1));
                edge.push((a - t.t0).abs() * 1000.0);
                dur.push((((b - a) - (t.t1 - t.t0)).abs()) * 1000.0);
            }
            edge.sort_by(f64::total_cmp);
            dur.sort_by(f64::total_cmp);
            println!("  {name:30} boundary error median {:4.0} ms, p90 {:4.0} ms | duration error median {:4.0} ms", percentile(&edge, 0.5), percentile(&edge, 0.9), percentile(&dur, 0.5));
        };
        println!("{text}");
        stats("just stretch the engine", &|t| t * span_scale);
        let unanchored = align(&eng, &truth, Vec::new())?;
        stats("warp, no word anchors", &|t| map_time(&unanchored, t));
        let anchored = align(&eng, &truth, points)?;
        stats("warp, word anchors", &|t| map_time(&anchored, t));
    }
    Ok(())
}

// ---- measurement

fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    Some(xs[xs.len() / 2])
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// The first three formants of one frame at `LOW_FS`. A frame on a fricative or a stray pole gives
/// an "F1" of 1900 Hz; those are refused rather than averaged in.
fn plausible_formants(frame: &[f64]) -> Option<[f64; 3]> {
    if frame.iter().map(|v| v * v).sum::<f64>() / (frame.len() as f64) < 1e-7 {
        return None;
    }
    let opts = FormantOpts { order: Some(12), pre_emph: Some(0.97), max_bw: Some(500.0), min_freq: Some(180.0) };
    let f = estimate_formants(frame, LOW_FS, opts);
    if f.len() < 3 {
        return None;
    }
    let (a, b, c) = (f[0].freq, f[1].freq, f[2].freq);
    ((180.0..=1000.0).contains(&a) && (600.0..=3000.0).contains(&b) && b - a > 200.0 && c <= 3600.0).then_some([a, b, c])
}

/// A sustained vowel's formants, from the steady middle of its longest loud stretch.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct HeldVowel {
    pub formants: [f64; 3],
    /// interquartile range of each formant over the frames used (Hz): how steady the vowel was
    pub spread: [f64; 3],
    pub f0: f64,
    pub steady_s: f64,
    pub frames: usize,
}

/// (file id, the vowel, the word to say, whether the vowel glides so only its start counts).
/// Each word has one vowel between consonants quiet enough to find it by loudness alone.
pub const WORDS: [(&str, &str, &str, bool); 13] = [
    ("fleece", "i", "seed", false),
    ("kit", "ɪ", "hid", false),
    ("face", "e", "paid", true),
    ("dress", "ɛ", "head", false),
    ("trap", "æ", "had", false),
    ("price", "a", "hide", true),
    ("start", "ɑ", "hard", false),
    ("lot", "ɒ", "cod", false),
    ("thought", "ɔ", "board", false),
    ("foot", "ʊ", "hood", false),
    ("goose", "u", "food", false),
    ("strut", "ʌ", "bud", false),
    ("nurse", "ɜ", "heard", false),
];

/// The vowel of one spoken word: the steady part of its longest loud stretch.
pub fn measure_word(x: &[f32], onset: bool) -> Option<HeldVowel> {
    let low: Vec<f64> = resample(x, FS, LOW_FS).iter().map(|&v| v as f64).collect();
    let hop = (0.01 * LOW_FS) as usize;
    let energy: Vec<f64> = low.chunks(hop).map(|c| c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).collect();
    if energy.len() < 20 {
        return None;
    }
    let mut sorted = energy.clone();
    sorted.sort_by(f64::total_cmp);
    let gate = percentile(&sorted, 0.95) * 0.1; // within 10 dB of the loud level
    let (mut best, mut start) = ((0, 0), None);
    for (i, &e) in energy.iter().chain(std::iter::once(&0.0)).enumerate() {
        match (e >= gate, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if i - s > best.1 - best.0 {
                    best = (s, i);
                }
                start = None;
            }
            _ => {}
        }
    }
    let run = best.1 - best.0;
    if (run as f64) * 0.01 < 0.1 {
        eprintln!("measure_word: longest loud stretch is only {} ms", run * 10);
        return None;
    }
    // past the consonant's transition and before the fade; a glide's start is its own target
    let (from, to) = if onset { (0.15, 0.45) } else { (0.25, 0.75) };
    let (a, b) = (best.0 * hop + (run as f64 * from) as usize * hop, best.0 * hop + (run as f64 * to) as usize * hop);
    let middle = &low[a..b.min(low.len())];
    let frame_len = (0.03 * LOW_FS) as usize;
    let step = (0.01 * LOW_FS) as usize;
    let mut found: Vec<[f64; 3]> = Vec::new();
    let (mut p, mut tried) = (0, 0);
    while p + frame_len <= middle.len() {
        tried += 1;
        if let Some(f) = plausible_formants(&middle[p..p + frame_len]) {
            found.push(f);
        } else if tried <= 3 {
            let opts = FormantOpts { order: Some(12), pre_emph: Some(0.97), max_bw: Some(500.0), min_freq: Some(180.0) };
            let raw: Vec<f64> = estimate_formants(&middle[p..p + frame_len], LOW_FS, opts).iter().take(4).map(|f| f.freq.round()).collect();
            eprintln!("measure_word: refused frame {tried}: {raw:?}");
        }
        p += step;
    }
    if found.len() < 3 {
        eprintln!("measure_word: stretch {} ms, {} of {tried} frames usable", run * 10, found.len());
        return None;
    }
    let (mut formants, mut spread) = ([0.0; 3], [0.0; 3]);
    for k in 0..3 {
        let mut v: Vec<f64> = found.iter().map(|f| f[k]).collect();
        v.sort_by(f64::total_cmp);
        formants[k] = percentile(&v, 0.5);
        spread[k] = percentile(&v, 0.75) - percentile(&v, 0.25);
    }
    Some(HeldVowel { formants, spread, f0: pitch_stats(middle).map_or(0.0, |p| p.median), steady_s: run as f64 * 0.01, frames: found.len() })
}

/// The engine's calibrated formants for a vowel, from its own table; `None` if it has no such vowel.
pub fn engine_formants(table: &serde_json::Value, body: &str, ipa: &str) -> Option<[f64; 3]> {
    let f = table.get(body)?.get(ipa)?.get("formants")?;
    Some([f.get(0)?.as_f64()?, f.get(1)?.as_f64()?, f.get(2)?.as_f64()?])
}

pub fn engine_vowel_table() -> serde_json::Value {
    std::fs::read_to_string(repo_path("crates/chimera/src/phonetics/vowelTargets.generated.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(serde_json::Value::Null)
}

/// Median of the first three formants across the vowel's middle, at 11 kHz where LPC behaves.
fn formants(low: &[f64], t0: f64, t1: f64) -> Option<[f64; 3]> {
    let frame_len = (0.03 * LOW_FS) as usize;
    let mut found: Vec<[f64; 3]> = Vec::new();
    for p in [0.35, 0.5, 0.65] {
        let c = ((t0 + (t1 - t0) * p) * LOW_FS) as isize;
        let start = c - frame_len as isize / 2;
        if start < 0 || start as usize + frame_len > low.len() {
            continue;
        }
        let frame = &low[start as usize..start as usize + frame_len];
        if let Some(f) = plausible_formants(frame) {
            found.push(f);
        }
    }
    if found.is_empty() {
        return None;
    }
    let mut out = [0.0; 3];
    for k in 0..3 {
        let mut v: Vec<f64> = found.iter().map(|f| f[k]).collect();
        out[k] = median(&mut v)?;
    }
    Some(out)
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PitchStats {
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
}

fn pitch_stats(low: &[f64]) -> Option<PitchStats> {
    let frame_len = (0.04 * LOW_FS) as usize;
    let hop = (0.01 * LOW_FS) as usize;
    let energies: Vec<f64> = low.chunks(hop).map(|c| c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).collect();
    let mut sorted = energies.clone();
    sorted.sort_by(f64::total_cmp);
    let gate = percentile(&sorted, 0.95) * 1e-3; // 30 dB below the loud level
    let opts = YinOpts { min_f0: Some(60.0), max_f0: Some(420.0), threshold: Some(0.2) };
    let mut f0s: Vec<f64> = Vec::new();
    let mut start = 0;
    while start + frame_len <= low.len() {
        let frame = &low[start..start + frame_len];
        if frame.iter().map(|v| v * v).sum::<f64>() / frame_len as f64 > gate {
            let p = yin_pitch(frame, LOW_FS, opts);
            if p.f0 > 0.0 {
                f0s.push(p.f0);
            }
        }
        start += hop;
    }
    if f0s.is_empty() {
        return None;
    }
    f0s.sort_by(f64::total_cmp);
    Some(PitchStats { median: percentile(&f0s, 0.5), p10: percentile(&f0s, 0.1), p90: percentile(&f0s, 0.9) })
}

// ---- the whole thing

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VowelRow {
    pub word: String,
    pub ipa: String,
    pub stress: u8,
    pub mine_ms: f64,
    pub engine_ms: f64,
    pub mine: Option<[f64; 3]>,
    pub engine: Option<[f64; 3]>,
    /// the engine with your timing, pitch and fitted vowels
    #[serde(default)]
    pub fitted: Option<[f64; 3]>,
}

/// Did fitting help? Same timing and pitch both sides, so only the vowel shapes differ. Each is the
/// mean |engine − you| / you (%), over the fitted vowels with a reading in all three recordings.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct FitGap {
    pub vowels: usize,
    pub before: [f64; 3],
    pub after: [f64; 3],
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub text: String,
    pub body: String,
    #[serde(default)]
    pub accent: String,
    /// what the engine was told to say
    #[serde(default)]
    pub phones: String,
    /// vowels with a believable reading in both voices
    #[serde(default)]
    pub formants_measured: usize,
    pub heard: String,
    pub words_total: usize,
    pub words_anchored: usize,
    /// how far the phoneme warp's word starts landed from Whisper's (median, ms)
    pub word_start_gap_ms: Option<f64>,
    pub mine_speech_s: f64,
    pub engine_speech_s: f64,
    pub mine_pitch: PitchStats,
    pub engine_pitch: PitchStats,
    /// mean |engine − mine| / mine (%) over the vowels measured in both
    pub formant_gap_pct: [f64; 3],
    pub vowels: Vec<VowelRow>,
    #[serde(default)]
    pub fit: Option<FitGap>,
}

impl Report {
    pub fn summary(&self) -> String {
        let fit = self.fit.map_or(String::new(), |f| format!("\nyour fitted vowels, over {} vowels: F1 {:.1}% to {:.1}%  F2 {:.1}% to {:.1}%  F3 {:.1}% to {:.1}%", f.vowels, f.before[0], f.after[0], f.before[1], f.after[1], f.before[2], f.after[2]));
        self.summary_base() + &fit
    }

    fn summary_base(&self) -> String {
        format!(
            "{} | body {} | words anchored {}/{} | warp vs Whisper word starts {} ms\nengine says: {}\nspeech {:.2} s (engine {:.2} s) | pitch {:.0} Hz, {:.0}-{:.0} (engine {:.0} Hz, {:.0}-{:.0})\nformant gap, engine vs you, over {}/{} vowels with a good reading: F1 {:.1}%  F2 {:.1}%  F3 {:.1}%\nWhisper heard: {}",
            self.accent,
            self.body,
            self.words_anchored,
            self.words_total,
            self.word_start_gap_ms.map_or("n/a".to_string(), |v| format!("{v:.0}")),
            self.phones,
            self.mine_speech_s,
            self.engine_speech_s,
            self.mine_pitch.median,
            self.mine_pitch.p10,
            self.mine_pitch.p90,
            self.engine_pitch.median,
            self.engine_pitch.p10,
            self.engine_pitch.p90,
            self.formants_measured,
            self.vowels.len(),
            self.formant_gap_pct[0],
            self.formant_gap_pct[1],
            self.formant_gap_pct[2],
            self.heard
        )
    }
}

/// `body` is "male", "female" or "auto" (decided by the recording's median pitch); `accent` is
/// "british" (southern) or "american" (the engine's own pronunciation).
pub fn analyze(wav_path: &str, text: &str, body: &str, accent: &str, out_dir: &str, status: &dyn Fn(&str)) -> Result<Report, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let mine = read_wav(wav_path)?;
    if mine.len() as f64 / FS > MAX_RECORDING_S {
        return Err(format!("recording is longer than {MAX_RECORDING_S} s; this tool is for single lines"));
    }
    if mine.len() < FS as usize / 2 {
        return Err("recording is too short".to_string());
    }
    let mine_low: Vec<f64> = resample(&mine, FS, LOW_FS).iter().map(|&v| v as f64).collect();
    let mine_pitch = pitch_stats(&mine_low).ok_or("couldn't find any voiced speech in the recording")?;
    let body = match body {
        "auto" => {
            if mine_pitch.median >= 160.0 {
                "female"
            } else {
                "male"
            }
        }
        b => b,
    };
    let default_f0 = if body == "female" { 205.0 } else { 112.0 };

    status("Rendering the engine's version");
    let g2p = load_g2p()?;
    let phrases = g2p.to_phrases(text);
    if phrases.is_empty() {
        return Err("that line has no words the engine can say".to_string());
    }
    // the fit is of southern British words, so it only applies to that accent, in its own body
    let fit = if accent == "british" { crate::fit::load().filter(|f| f.body == body) } else { None };
    let phrases = if accent == "british" { crate::british::southern_british(&g2p, &phrases, body, default_f0, fit.as_ref().is_some_and(|f| f.get("ɜ").is_some()))? } else { phrases };
    let said: Vec<String> = phrases.iter().flat_map(|p| p.words.iter().cloned()).collect();
    let phrase_inputs = inputs(&phrases);
    let eng_plan = plan(&phrase_inputs, body, default_f0);
    let eng = render(&eng_plan, body, default_f0);
    write_wav(&format!("{out_dir}/engine.wav"), &eng)?;
    let spans = seg_spans(&phrases, &eng_plan.segments)?;

    status(if whisper_model_cached() { "Whisper is timing your words" } else { "Downloading Whisper's model (about 0.5 GB, first run only), then timing your words" });
    // Whisper's loader takes only 16-bit PCM, so hand it a normalized copy of whatever came in
    let whisper_input = format!("{out_dir}/whisper-input.wav");
    write_wav(&whisper_input, &mine)?;
    let heard = run_whisper(&whisper_input, &format!("{out_dir}/whisper.json"));
    let _ = std::fs::remove_file(&whisper_input);
    let heard = heard?;
    let pairs = match_words(&said, &heard);

    status("Aligning your phonemes to the engine's");
    let word_span = |w: usize| -> Option<(f64, f64)> {
        let mut it = spans.iter().filter(|s| s.word == w);
        let first = it.next()?;
        let last = it.last().unwrap_or(first);
        Some((eng_plan.segments[first.first].t0, eng_plan.segments[last.first + last.count - 1].t1))
    };
    // guide points (engine time, recording time) from the words both sides agree on
    let mut points: Vec<(f64, f64)> = Vec::new();
    for &(w, h) in &pairs {
        let Some((e0, e1)) = word_span(w) else { continue };
        let (m0, m1) = (heard[h].start, heard[h].end);
        let last = points.last().copied().unwrap_or((f64::NEG_INFINITY, f64::NEG_INFINITY));
        if e0 > last.0 && m0 >= last.1 && e1 > e0 && m1 > m0 {
            points.push((e0, m0));
            points.push((e1, m1));
        }
    }
    let path = align(&eng, &mine, points)?;
    let to_mine = |t: f64| map_time(&path, t);

    // every planned segment's boundaries, in recording time, never running backwards
    let mut mine_times: Vec<(f64, f64)> = Vec::with_capacity(eng_plan.segments.len());
    let mut prev = 0.0f64;
    for s in &eng_plan.segments {
        let a = to_mine(s.t0).max(prev);
        let b = to_mine(s.t1).max(a);
        mine_times.push((a, b));
        prev = b;
    }

    let mut gaps: Vec<f64> = pairs
        .iter()
        .filter_map(|&(w, h)| {
            let (e0, _) = word_span(w)?;
            Some((to_mine(e0) - heard[h].start).abs() * 1000.0)
        })
        .collect();
    let word_start_gap_ms = median(&mut gaps);

    status("Measuring pitch and vowels");
    let eng_low: Vec<f64> = resample(&eng, FS, LOW_FS).iter().map(|&v| v as f64).collect();
    let engine_pitch = pitch_stats(&eng_low).unwrap_or_default();
    let mut vowels: Vec<VowelRow> = Vec::new();
    let (mut sums, mut n) = ([0.0f64; 3], 0usize);
    for span in &spans {
        for k in span.first..span.first + span.count {
            let s = &eng_plan.segments[k];
            if s.kind != "vowel" {
                continue;
            }
            let (a, b) = mine_times[k];
            let m = formants(&mine_low, a, b);
            let e = formants(&eng_low, s.t0, s.t1);
            if let (Some(m), Some(e)) = (m, e) {
                for c in 0..3 {
                    sums[c] += 100.0 * (e[c] - m[c]).abs() / m[c];
                }
                n += 1;
            }
            vowels.push(VowelRow {
                word: said.get(span.word).cloned().unwrap_or_default(),
                ipa: s.ipa.clone(),
                stress: s.stress.unwrap_or(0),
                mine_ms: (b - a) * 1000.0,
                engine_ms: (s.t1 - s.t0) * 1000.0,
                mine: m,
                engine: e,
                fitted: None,
            });
        }
    }
    let formant_gap_pct = if n > 0 { sums.map(|v| v / n as f64) } else { [f64::NAN; 3] };

    status("Rendering the engine with your timing and pitch");
    let mut timed = phrase_inputs.clone();
    for span in &spans {
        let (a, _) = mine_times[span.first];
        let (_, b) = mine_times[span.first + span.count - 1];
        let vowel = eng_plan.segments[span.first].kind == "vowel";
        timed[span.phrase].segments[span.seg].dur = Some((b - a).max(if vowel { 0.04 } else { 0.03 }));
    }
    for pi in 0..phrases.len().saturating_sub(1) {
        let last = spans.iter().rfind(|s| s.phrase == pi);
        let next = spans.iter().find(|s| s.phrase == pi + 1);
        if let (Some(l), Some(n)) = (last, next) {
            let end = mine_times[l.first + l.count - 1].1;
            let start = mine_times[n.first].0;
            timed[pi].pause = Some((start - end - 0.06).max(0.03)); // the next phrase's breath lead
        }
    }
    let my_f0 = mine_pitch.median.clamp(70.0, 350.0);
    let timed_plan = plan(&timed, body, my_f0);
    let timed_audio = render(&timed_plan, body, my_f0);
    write_wav(&format!("{out_dir}/engine-my-timing.wav"), &timed_audio)?;

    let mut fit_gap = None;
    let fitted_path = format!("{out_dir}/engine-my-vowels.wav");
    let _ = std::fs::remove_file(&fitted_path);
    if let Some(fit) = &fit {
        status("Rendering the engine with your fitted vowels");
        let mut fitted_plan = timed_plan.clone();
        crate::fit::apply(&mut fitted_plan, fit);
        let fitted_audio = render(&fitted_plan, body, my_f0);
        write_wav(&fitted_path, &fitted_audio)?;
        let before_low: Vec<f64> = resample(&timed_audio, FS, LOW_FS).iter().map(|&v| v as f64).collect();
        let after_low: Vec<f64> = resample(&fitted_audio, FS, LOW_FS).iter().map(|&v| v as f64).collect();
        let (mut before, mut after, mut count) = ([0.0f64; 3], [0.0f64; 3], 0usize);
        for (row, s) in vowels.iter_mut().zip(timed_plan.segments.iter().filter(|s| s.kind == "vowel")) {
            let a = formants(&after_low, s.t0, s.t1);
            row.fitted = a;
            if fit.get(crate::fit::fitted_label(&s.ipa)).is_none() {
                continue;
            }
            if let (Some(m), Some(b), Some(a)) = (row.mine, formants(&before_low, s.t0, s.t1), a) {
                for c in 0..3 {
                    before[c] += 100.0 * (b[c] - m[c]).abs() / m[c];
                    after[c] += 100.0 * (a[c] - m[c]).abs() / m[c];
                }
                count += 1;
            }
        }
        if count > 0 {
            fit_gap = Some(FitGap { vowels: count, before: before.map(|v| v / count as f64), after: after.map(|v| v / count as f64) });
        }
    }
    for row in vowels.iter_mut() {
        row.ipa = crate::fit::fitted_label(&row.ipa).to_string();
    }

    let speech_span = |first: f64, last: f64| (last - first).max(0.0);
    let report = Report {
        text: text.to_string(),
        body: body.to_string(),
        accent: accent.to_string(),
        phones: crate::british::phones(&phrases),
        formants_measured: n,
        heard: heard.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" "),
        words_total: said.len(),
        words_anchored: pairs.len(),
        word_start_gap_ms,
        mine_speech_s: speech_span(mine_times[0].0, mine_times[mine_times.len() - 1].1),
        engine_speech_s: speech_span(eng_plan.segments[0].t0, eng_plan.segments[eng_plan.segments.len() - 1].t1),
        mine_pitch,
        engine_pitch,
        formant_gap_pct,
        vowels,
        fit: fit_gap,
    };
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(format!("{out_dir}/report.json"), json).map_err(|e| format!("{out_dir}: {e}"))?;
    Ok(report)
}

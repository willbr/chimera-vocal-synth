//! Builds a voice of filter tables from recordings, and turns text into the synth's events. This
//! side runs on the PC and may use floating point; only `n64.rs` is console code. (A console
//! build would need its own integer port of the text-to-events step.)

use crate::n64::{self, Event, FRAME_BYTES, ORDER, OUT_RATE, PULSE_LEN, Params, Stats, Voice};
use crate::pipeline::{self, FS, inputs, plan, read_wav, resample, seg_spans, write_wav};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::Intonation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::Mutex;

const HOP: usize = 160; // 10 ms at 16 kHz
const WINDOW: usize = 400;
const PRE: f64 = 0.95;
const SILENCE: &str = "#";
const FS_OUT: f64 = OUT_RATE as f64;
/// Longest stretch of frames one join may keep, so a bad alignment can't store seconds of audio.
const MAX_JOIN_FRAMES: usize = 60;

/// A join's place in the voice file.
struct Entry {
    a: u8,
    b: u8,
    offset: u32,
    frames: u16,
    join: u16,
}

// ---- frame analysis

#[derive(Clone)]
struct Frame {
    lsf: [u8; ORDER],
    gain: u8,
    voiced: u8,
}

fn levinson(r: &[f64]) -> Option<([f64; ORDER + 1], f64)> {
    let mut a = [0.0; ORDER + 1];
    a[0] = 1.0;
    let mut err = r[0];
    if err <= 0.0 {
        return None;
    }
    for i in 1..=ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        let prev = a;
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            return None;
        }
    }
    Some((a, err))
}

/// The roots of the sum and difference polynomials on the unit circle, in order.
fn to_lsf(a: &[f64; ORDER + 1]) -> Option<[f64; ORDER]> {
    let m = ORDER;
    let (mut p, mut q) = ([0.0; ORDER + 1], [0.0; ORDER + 1]);
    // symmetric and antisymmetric parts with their trivial roots divided out
    let (mut last_p, mut last_q) = (0.0, 0.0);
    for i in 0..=m {
        let (head, tail) = (a[i], if i == 0 { 0.0 } else { a[m + 1 - i] });
        p[i] = head + tail - last_p;
        q[i] = head - tail + last_q;
        last_p = p[i];
        last_q = q[i];
    }
    let eval = |c: &[f64; ORDER + 1], w: f64| -> f64 {
        let mut s = c[m / 2];
        for k in 0..m / 2 {
            s += 2.0 * c[k] * (((m / 2 - k) as f64) * w).cos();
        }
        s
    };
    let mut roots = Vec::with_capacity(m);
    for c in [&p, &q] {
        let steps = 2048;
        let mut prev_w = 0.0;
        let mut prev_v = eval(c, prev_w);
        for s in 1..=steps {
            let w = PI * s as f64 / steps as f64;
            let v = eval(c, w);
            if prev_v * v < 0.0 {
                let (mut lo, mut hi, mut vlo) = (prev_w, w, prev_v);
                for _ in 0..30 {
                    let mid = 0.5 * (lo + hi);
                    let vm = eval(c, mid);
                    if vm * vlo <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        vlo = vm;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            prev_w = w;
            prev_v = v;
        }
    }
    if roots.len() != m {
        return None;
    }
    roots.sort_by(f64::total_cmp);
    let mut out = [0.0; ORDER];
    out.copy_from_slice(&roots);
    Some(out)
}

fn lpc_of(frame: &[f64], window: &[f64]) -> Option<([f64; ORDER + 1], f64)> {
    let w: Vec<f64> = frame.iter().zip(window).map(|(a, b)| a * b).collect();
    let mut r = vec![0.0; ORDER + 1];
    for lag in 0..=ORDER {
        r[lag] = w[..w.len() - lag].iter().zip(&w[lag..]).map(|(a, b)| a * b).sum();
    }
    r[0] = r[0] * 1.0001 + 1e-3;
    for (lag, value) in r.iter_mut().enumerate().skip(1) {
        *value *= (-0.5 * (2.0 * PI * 40.0 * lag as f64 / FS_OUT).powi(2)).exp();
    }
    levinson(&r)
}

struct Analysis {
    frames: Vec<Frame>,
    filters: Vec<Option<[f64; ORDER + 1]>>,
    pre: Vec<f64>,
    f0: Vec<f64>,
}

fn analyze(x16: &[f32]) -> Analysis {
    let raw: Vec<f64> = x16.iter().map(|&v| v as f64 * 32768.0).collect();
    let mut pre = vec![0.0; raw.len()];
    for i in 1..raw.len() {
        pre[i] = raw[i] - PRE * raw[i - 1];
    }
    let window: Vec<f64> = (0..WINDOW).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (WINDOW - 1) as f64).cos()).collect();
    let energy: f64 = window.iter().map(|w| w * w).sum();
    let yin = YinOpts { min_f0: Some(60.0), max_f0: Some(420.0), threshold: Some(0.25) };
    let (mut frames, mut filters, mut f0s) = (Vec::new(), Vec::new(), Vec::new());
    let quiet_lsf = {
        let mut q = [0u8; ORDER];
        for (k, v) in q.iter_mut().enumerate() {
            *v = (((k + 1) * 256) / (ORDER + 1)) as u8;
        }
        q
    };
    let mut prev = quiet_lsf;
    let mut start = 0;
    while start + WINDOW <= pre.len() {
        let chunk = &pre[start..start + WINDOW];
        let solved = lpc_of(chunk, &window);
        let (mut gain, mut lsf, mut filter) = (0u8, prev, None);
        if let Some((a, err)) = solved {
            let rms = (err / energy).max(0.0).sqrt();
            gain = (16.0 * (rms + 1.0).log2()).round().clamp(0.0, 255.0) as u8;
            if let Some(roots) = to_lsf(&a) {
                let mut floor = 1i32;
                for k in 0..ORDER {
                    let b = ((roots[k] / PI * 256.0) - 0.5).round() as i32;
                    let value = b.clamp(floor, 254 - (ORDER - 1 - k) as i32);
                    lsf[k] = value as u8;
                    floor = value + 1;
                }
                filter = Some(a);
            }
        }
        prev = lsf;
        let p = yin_pitch(&raw[start..start + WINDOW.min(raw.len() - start)], FS_OUT, yin);
        let voiced = if p.f0 > 0.0 && gain > 70 { ((1.0 - p.aperiodicity) * 15.0).round().clamp(0.0, 15.0) as u8 } else { 0 };
        f0s.push(if voiced > 0 { p.f0 } else { 0.0 });
        frames.push(Frame { lsf, gain, voiced });
        filters.push(filter);
        start += HOP;
    }
    Analysis { frames, filters, pre, f0: f0s }
}

/// One pitch period of the residual, averaged over the speaker's voiced frames.
fn average_pulse(analyses: &[Analysis]) -> Vec<i16> {
    let mut sum = vec![0.0f64; PULSE_LEN];
    let mut count = 0usize;
    for an in analyses {
        for (i, frame) in an.frames.iter().enumerate() {
            let (Some(a), f0) = (&an.filters[i], an.f0[i]) else { continue };
            if frame.voiced < 12 || f0 <= 0.0 || i % 3 != 0 {
                continue;
            }
            let start = i * HOP;
            let t = (FS_OUT / f0) as usize;
            if start < ORDER + t || start + 2 * t + ORDER >= an.pre.len() {
                continue;
            }
            let residual: Vec<f64> = (start..start + 2 * t).map(|n| (0..=ORDER).map(|k| a[k] * an.pre[n - k]).sum()).collect();
            let peak = (0..residual.len() - t).max_by(|&x, &y| residual[x].abs().total_cmp(&residual[y].abs())).unwrap_or(0);
            let period = &residual[peak..peak + t];
            let rms = (period.iter().map(|v| v * v).sum::<f64>() / t as f64).sqrt().max(1e-6);
            let sign = if period[0] < 0.0 { -1.0 } else { 1.0 };
            for (k, s) in sum.iter_mut().enumerate() {
                // the main excitation sits at the pulse's centre
                let pos = ((k + PULSE_LEN / 2) % PULSE_LEN) as f64 / PULSE_LEN as f64 * t as f64;
                let (a0, frac) = (pos.floor() as usize, pos.fract());
                let v = period[a0 % t] * (1.0 - frac) + period[(a0 + 1) % t] * frac;
                *s += sign * v / rms;
            }
            count += 1;
        }
    }
    if count == 0 {
        // no voiced speech to learn from: a plain impulse-like pulse
        for (k, s) in sum.iter_mut().enumerate() {
            *s = (-((k as f64 - 64.0) / 5.0).powi(2)).exp();
        }
    }
    let mean = sum.iter().sum::<f64>() / PULSE_LEN as f64;
    let centred: Vec<f64> = sum.iter().map(|v| v - mean).collect();
    let rms = (centred.iter().map(|v| v * v).sum::<f64>() / PULSE_LEN as f64).sqrt().max(1e-9);
    centred.iter().map(|v| (v / rms * 4096.0).round().clamp(-32768.0, 32767.0) as i16).collect()
}

// ---- building a voice

pub struct Take {
    pub wav: String,
    pub text: String,
}

struct Candidate {
    a: String,
    b: String,
    frames: Vec<Frame>,
    join: usize,
}

type PhoneTimes = Vec<(String, f64, f64, usize)>;

/// One stretch the aligner will find: the labels it may use, and the planned segments it covers.
struct Unit {
    options: Vec<&'static str>,
    first: usize,
    count: usize,
}

fn is_vowel_label(ipa: &str) -> bool {
    matches!(ipa, "i" | "ɪ" | "eɪ" | "ɛ" | "æ" | "aɪ" | "aʊ" | "ɔɪ" | "ɑ" | "ɒ" | "ɔ" | "ʊ" | "u" | "ʌ" | "ə" | "ɐ")
}

/// Our sound as the aligner's British model names it. `next` is the following sound in the word:
/// the model has separate palatal and aspirated consonants, chosen by what comes after.
fn aligner_labels(ipa: &str, next: Option<&str>) -> Option<Vec<&'static str>> {
    let vowel = next.is_some_and(is_vowel_label);
    let front = matches!(next, Some("i" | "ɪ" | "j"));
    let one = |s: &'static str| Some(vec![s]);
    match ipa {
        "i" => one("iː"),
        "ɪ" => one("ɪ"),
        "eɪ" => one("ej"),
        "ɛ" => Some(vec!["ɛ", "ɛː"]),
        "æ" => one("a"),
        "aɪ" => one("aj"),
        "aʊ" => one("aw"),
        "ɔɪ" => one("ɔj"),
        "ɑ" => one("ɑː"),
        "ɒ" => one("ɒ"),
        "ɔ" => one("ɒː"),
        "ʊ" => one("ʊ"),
        "u" => one("ʉː"),
        "ʌ" | "ɐ" => one("ɐ"),
        // our NURSE is written ə, so either may be what was said
        "ə" => Some(vec!["ə", "ɜː"]),
        "p" => one(if front { "pʲ" } else if vowel { "pʰ" } else { "p" }),
        "t" if front => one("tʲ"),
        "t" if vowel => one("tʰ"),
        // a Southampton t at the end of a syllable is often a glottal stop
        "t" => Some(vec!["t", "ʔ"]),
        "k" => one(if front { "cʰ" } else if vowel { "kʰ" } else { "k" }),
        "b" => one(if front { "bʲ" } else { "b" }),
        "d" => one(if front { "dʲ" } else { "d" }),
        "g" => one(if front { "ɟ" } else { "ɡ" }),
        "f" => one(if front { "fʲ" } else { "f" }),
        "v" => one(if front { "vʲ" } else { "v" }),
        "m" => one(if front { "mʲ" } else { "m" }),
        "n" => one(if front { "ɲ" } else { "n" }),
        "l" => one(if front { "ʎ" } else { "l" }),
        "ɫ" => one("ɫ"),
        "s" => one("s"),
        "z" => one("z"),
        "ʃ" => one("ʃ"),
        "ʒ" => one("ʒ"),
        "h" => one("h"),
        "j" => one("j"),
        "w" => one("w"),
        "ɹ" => one("ɹ"),
        "ŋ" => one("ŋ"),
        "ð" => one("ð"),
        "θ" => one("θ"),
        "tʃ" => one("tʃ"),
        "dʒ" => one("dʒ"),
        _ => None,
    }
}

/// Every way of saying a word, capped: variants multiply.
fn pronunciations(units: &[Unit]) -> Vec<String> {
    let mut all = vec![String::new()];
    for u in units {
        let options = if all.len() * u.options.len() > 64 { &u.options[..1] } else { &u.options[..] };
        all = all.iter().flat_map(|p| options.iter().map(move |o| if p.is_empty() { o.to_string() } else { format!("{p} {o}") })).collect();
    }
    all
}

/// One label per planned segment. A diphthong's glide is its own sound: without this the end of
/// "days" is filed as the "ee" of "please", and either word can borrow the other's piece.
fn phone_labels(phrases: &[chimera::phonetics::english::Phrase], planned: &chimera::phonetics::Plan, spans: &[pipeline::SegSpan]) -> Vec<String> {
    let mut labels = vec![String::new(); planned.segments.len()];
    let ipa = |n: usize| phrases[spans[n].phrase].segments[spans[n].seg].ipa.as_str();
    let in_word = |a: usize, b: usize| spans.get(a).is_some_and(|x| spans.get(b).is_some_and(|y| x.word == y.word));
    for (n, span) in spans.iter().enumerate() {
        let goat = (ipa(n) == "ə" && in_word(n, n + 1) && ipa(n + 1) == "ʊ") || (ipa(n) == "ʊ" && n > 0 && in_word(n - 1, n) && ipa(n - 1) == "ə");
        for part in 0..span.count {
            let base = &planned.segments[span.first + part].ipa;
            let glide = goat || (span.count == 2 && part == 1);
            labels[span.first + part] = if glide { format!("{base}'") } else { base.clone() };
        }
    }
    labels
}

struct Planned {
    words: Vec<Vec<Unit>>,
    segments: Vec<(String, f64, usize)>,
}

/// The sounds of a take in the order our own front end would say them, grouped by word.
fn plan_take(g2p: &EnglishG2P, text: &str) -> Result<Planned, String> {
    let phrases = g2p.to_phrases(text);
    if phrases.is_empty() {
        return Err("no words".to_string());
    }
    let phrases = crate::british::southern_british(g2p, &phrases, "male", 112.0, false)?;
    let planned = plan(&inputs(&phrases), "male", 112.0);
    let spans = seg_spans(&phrases, &planned.segments)?;
    let labels = phone_labels(&phrases, &planned, &spans);
    let segments = spans.iter().flat_map(|sp| (sp.first..sp.first + sp.count).map(|k| (labels[k].clone(), planned.segments[k].t1 - planned.segments[k].t0, sp.phrase))).collect();
    let word_count = spans.last().map_or(0, |s| s.word + 1);
    let mut words = Vec::with_capacity(word_count);
    for w in 0..word_count {
        let inside: Vec<_> = spans.iter().filter(|s| s.word == w).collect();
        let ipa = |i: usize| inside.get(i).map(|s| phrases[s.phrase].segments[s.seg].ipa.as_str());
        let mut units = Vec::with_capacity(inside.len());
        let mut i = 0;
        while i < inside.len() {
            let here = ipa(i).unwrap_or("");
            // the British front end writes GOAT as two sounds; the model has it as one
            if here == "ə" && ipa(i + 1) == Some("ʊ") {
                units.push(Unit { options: vec!["əw"], first: inside[i].first, count: inside[i].count + inside[i + 1].count });
                i += 2;
                continue;
            }
            let options = aligner_labels(here, ipa(i + 1)).ok_or_else(|| format!("no aligner label for {here}"))?;
            units.push(Unit { options, first: inside[i].first, count: inside[i].count });
            i += 1;
        }
        if !units.is_empty() {
            words.push(units);
        }
    }
    Ok(Planned { words, segments })
}

fn write_16k(path: &str, x: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec { channels: 1, sample_rate: OUT_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| format!("{path}: {e}"))?;
    for &s in x {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).map_err(|e| format!("{path}: {e}"))?;
    }
    w.finalize().map_err(|e| format!("{path}: {e}"))
}

/// One take's result from the aligner, back onto our own sounds.
fn read_alignment(path: &str, planned: &Planned) -> Result<PhoneTimes, String> {
    let text = std::fs::read_to_string(path).map_err(|_| "the aligner couldn't place this take".to_string())?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let entries = json["tiers"]["phones"]["entries"].as_array().ok_or("no phones in the aligner's output")?;
    let found: Vec<(f64, f64)> = entries.iter().filter(|e| !matches!(e[2].as_str(), None | Some("" | "sil" | "spn" | "<eps>"))).filter_map(|e| Some((e[0].as_f64()?, e[1].as_f64()?))).collect();
    let units: Vec<&Unit> = planned.words.iter().flatten().collect();
    if found.len() != units.len() {
        return Err(format!("the aligner found {} sounds, expected {}", found.len(), units.len()));
    }
    let mut times = Vec::with_capacity(planned.segments.len());
    let mut group = 0usize;
    let (mut prev_end, mut prev_phrase) = (None::<f64>, 0usize);
    for (unit, &(start, end)) in units.iter().zip(&found) {
        let total: f64 = planned.segments[unit.first..unit.first + unit.count].iter().map(|s| s.1).sum::<f64>().max(1e-6);
        let phrase = planned.segments[unit.first].2;
        // a real pause splits the take, so no join is cut across silence
        if phrase != prev_phrase || prev_end.is_some_and(|p| start - p >= 0.06) {
            group += 1;
        }
        let mut t = start;
        for s in &planned.segments[unit.first..unit.first + unit.count] {
            let next = t + (end - start) * s.1 / total;
            times.push((s.0.clone(), t, next, group));
            t = next;
        }
        prev_end = Some(end);
        prev_phrase = phrase;
    }
    Ok(times)
}

/// Run Montreal Forced Aligner over the takes, once. It only ever runs here, on the PC.
fn mfa_align(g2p: &EnglishG2P, takes: &[Take], work: &str) -> Result<Vec<Result<PhoneTimes, String>>, String> {
    let home = std::env::var("USERPROFILE").map_err(|_| "USERPROFILE isn't set".to_string())?;
    let root = format!("{home}/.cache/chimera-asr/mfa");
    let micromamba = format!("{root}/micromamba.exe");
    if !std::path::Path::new(&micromamba).exists() {
        return Err(format!("Montreal Forced Aligner isn't installed (expected it in {root})"));
    }
    let (corpus, output) = (format!("{work}/corpus"), format!("{work}/aligned"));
    std::fs::create_dir_all(&corpus).map_err(|e| format!("{corpus}: {e}"))?;
    let mut dictionary = String::new();
    let mut planned = Vec::with_capacity(takes.len());
    for (i, take) in takes.iter().enumerate() {
        let plan_result = plan_take(g2p, &take.text);
        if let Ok(p) = &plan_result {
            let x = read_wav(&take.wav)?;
            write_16k(&format!("{corpus}/t{i}.wav"), &resample(&x, FS, FS_OUT))?;
            let tokens: Vec<String> = (0..p.words.len()).map(|w| format!("t{i}w{w}")).collect();
            std::fs::write(format!("{corpus}/t{i}.lab"), tokens.join(" ")).map_err(|e| e.to_string())?;
            // each spoken word is its own dictionary entry, so the aligner must follow our sounds
            for (token, units) in tokens.iter().zip(&p.words) {
                for pron in pronunciations(units) {
                    dictionary.push_str(&format!("{token}\t{pron}\n"));
                }
            }
        }
        planned.push(plan_result);
    }
    let dict_path = format!("{work}/words.dict");
    std::fs::write(&dict_path, dictionary).map_err(|e| format!("{dict_path}: {e}"))?;
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8).to_string();
    let mut cmd = std::process::Command::new(&micromamba);
    cmd.args(["run", "-r", &format!("{root}/mamba"), "-p", &format!("{root}/env"), "mfa", "align", &corpus, &dict_path, "english_mfa", &output]);
    cmd.args(["--clean", "--single_speaker", "--output_format", "json", "--beam", "100", "--retry_beam", "400", "-j", &jobs, "--temporary_directory", &format!("{work}/tmp")]);
    cmd.env("MFA_ROOT_DIR", format!("{root}/models")).env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let result = cmd.output().map_err(|e| format!("couldn't start the aligner: {e}"))?;
    if !result.status.success() {
        let (out, err) = (String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        let tail: Vec<&str> = out.lines().chain(err.lines()).rev().take(12).collect();
        return Err(format!("the aligner failed:\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n")));
    }
    Ok(planned.iter().enumerate().map(|(i, p)| read_alignment(&format!("{output}/t{i}.json"), p.as_ref().map_err(|e| e.clone())?)).collect())
}

/// Vowels, nasals and approximants are voiced whatever the pitch tracker said: it gave up on many
/// loud vowel frames, and a frame stored as unvoiced plays as pure noise. A little noise stays.
fn mark_voiced(an: &mut Analysis, times: &[(String, f64, f64, usize)]) {
    let count = an.frames.len() as isize;
    if count == 0 {
        return;
    }
    let frame = |t: f64| (((t - 0.0125) / 0.01).round() as isize).clamp(0, count - 1) as usize;
    for (ipa, t0, t1, _) in times {
        if matches!(ipa.trim_end_matches('\''), "a" | "e" | "i" | "u" | "æ" | "ɑ" | "ɒ" | "ɔ" | "ə" | "ɛ" | "ɪ" | "ʊ" | "ʌ" | "ɐ" | "m" | "n" | "ŋ" | "l" | "ɫ" | "ɹ" | "w" | "j") {
            for i in frame(*t0)..=frame(*t1) {
                let f = &mut an.frames[i];
                if f.gain >= 70 {
                    f.voiced = f.voiced.max(14);
                }
            }
        }
    }
}

fn analysis_of(take: &Take) -> Result<Analysis, String> {
    let mine = read_wav(&take.wav)?;
    Ok(analyze(&resample(&mine, FS, FS_OUT)))
}

fn align_take(take: &Take, times: &Result<PhoneTimes, String>) -> Result<(Vec<Candidate>, Analysis), String> {
    let times = times.as_ref().map_err(|e| format!("{}: {e}", take.wav))?;
    let mut analysis = analysis_of(take)?;
    mark_voiced(&mut analysis, times);
    let dur = analysis.frames.len() as f64 * 0.01;
    // frame i is centred half a window (12.5 ms) after its start
    let frame_at = |t: f64| (((t - 0.0125) / 0.01).round() as isize).clamp(0, analysis.frames.len() as isize - 1) as usize;
    let mut out: Vec<Candidate> = Vec::new();
    let mut harvest = |a: &str, b: &str, t0: f64, join_t: f64, t1: f64| {
        let (i0, j, i1) = (frame_at(t0), frame_at(join_t), frame_at(t1));
        if j <= i0 || i1 <= j || i1 - i0 > MAX_JOIN_FRAMES {
            return;
        }
        out.push(Candidate { a: a.to_string(), b: b.to_string(), frames: analysis.frames[i0..i1].to_vec(), join: j - i0 });
    };
    let mid = |t: &(String, f64, f64, usize)| 0.5 * (t.1 + t.2);
    if let (Some(first), Some(last)) = (times.first(), times.last()) {
        harvest(SILENCE, &first.0, (first.1 - 0.06).max(0.0), first.1, mid(first));
        harvest(&last.0, SILENCE, mid(last), last.2, (last.2 + 0.06).min(dur));
    }
    for w in times.windows(2) {
        if w[0].3 != w[1].3 {
            // a pause between phrases: into and out of silence, not a join across it
            harvest(&w[0].0, SILENCE, mid(&w[0]), w[0].2, (w[0].2 + 0.06).min(dur));
            harvest(SILENCE, &w[1].0, (w[1].1 - 0.06).max(0.0), w[1].1, mid(&w[1]));
        } else {
            harvest(&w[0].0, &w[1].0, mid(&w[0]), w[0].2, mid(&w[1]));
        }
    }
    Ok((out, analysis))
}

const SHAPE_POINTS: usize = 16;

/// An example's filter and loudness at evenly spaced points, half each side of its join. Near
/// silent frames count for little (their filter is noise); low resonances count most.
fn shape(c: &Candidate) -> Vec<f64> {
    let mut out = Vec::with_capacity(SHAPE_POINTS * (ORDER + 1));
    let side = |range: &[Frame], out: &mut Vec<f64>| {
        for p in 0..SHAPE_POINTS / 2 {
            let f = &range[p * range.len() / (SHAPE_POINTS / 2)];
            let weight = if f.gain >= 70 { 1.0 } else { 0.2 };
            for k in 0..ORDER {
                out.push(f.lsf[k] as f64 * weight * if k < 10 { 1.0 } else { 0.5 });
            }
            out.push(f.gain as f64 * weight * 0.5);
        }
    };
    side(&c.frames[..c.join], &mut out);
    side(&c.frames[c.join..], &mut out);
    out
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BuildReport {
    pub takes: usize,
    pub phones: usize,
    pub joins: usize,
    pub frames: usize,
    pub rom_bytes: usize,
    pub failed: usize,
    pub single_example_joins: usize,
}

/// Build a voice file from takes. Whisper runs once for all of them.
pub fn build(takes: &[Take], work_dir: &str, out_path: &str, status: &(dyn Fn(&str) + Sync)) -> Result<BuildReport, String> {
    if takes.is_empty() {
        return Err("no recordings yet".to_string());
    }
    std::fs::create_dir_all(work_dir).map_err(|e| format!("{work_dir}: {e}"))?;
    let g2p = pipeline::load_g2p()?;
    status("Finding every sound in the recordings");
    let times = mfa_align(&g2p, takes, work_dir)?;
    status("Measuring the recordings");
    let next = Mutex::new(0usize);
    let results: Mutex<Vec<Result<(Vec<Candidate>, Analysis), String>>> = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(12);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = {
                        let mut n = next.lock().unwrap();
                        let i = *n;
                        *n += 1;
                        i
                    };
                    if i >= takes.len() {
                        break;
                    }
                    let r = align_take(&takes[i], &times[i]);
                    results.lock().unwrap().push(r);
                }
            });
        }
    });
    status("Choosing the best piece of each join");
    let (mut candidates, mut analyses, mut failed) = (Vec::new(), Vec::new(), 0);
    for r in results.into_inner().unwrap() {
        match r {
            Ok((c, a)) => {
                candidates.extend(c);
                analyses.push(a);
            }
            Err(e) => {
                eprintln!("take failed: {e}");
                failed += 1;
            }
        }
    }
    if candidates.is_empty() {
        return Err("no usable sounds found in the recordings".to_string());
    }
    let pulse = average_pulse(&analyses);

    // the typical filter of each phone, from the sides of its joins
    let mut phones: Vec<String> = candidates.iter().flat_map(|c| [c.a.clone(), c.b.clone()]).collect::<HashSet<_>>().into_iter().collect();
    phones.sort();
    if phones.len() > 255 {
        return Err("too many different sounds".to_string());
    }
    let id: HashMap<String, u8> = phones.iter().enumerate().map(|(i, p)| (p.clone(), i as u8)).collect();
    let mut totals: HashMap<u8, ([f64; ORDER], usize)> = HashMap::new();
    let mut add = |phone: u8, f: &Frame| {
        let e = totals.entry(phone).or_insert(([0.0; ORDER], 0));
        for k in 0..ORDER {
            e.0[k] += f.lsf[k] as f64;
        }
        e.1 += 1;
    };
    for c in &candidates {
        add(id[&c.a], &c.frames[0]);
        add(id[&c.b], &c.frames[c.frames.len() - 1]);
    }
    let mean: HashMap<u8, [f64; ORDER]> = totals.iter().map(|(p, (s, n))| (*p, s.map(|v| v / *n as f64))).collect();
    let distance = |f: &Frame, m: &[f64; ORDER]| -> f64 { (0..ORDER).map(|k| (f.lsf[k] as f64 - m[k]).abs()).sum() };
    let mut groups: BTreeMap<(u8, u8), Vec<Candidate>> = BTreeMap::new();
    for c in candidates {
        groups.entry((id[&c.a], id[&c.b])).or_default().push(c);
    }
    let single_example_joins = groups.values().filter(|g| g.len() == 1).count();
    let mut best: BTreeMap<(u8, u8), (f64, Candidate)> = BTreeMap::new();
    for ((a, b), group) in groups {
        let shapes: Vec<Vec<f64>> = group.iter().map(shape).collect();
        let typical: Vec<f64> = (0..shapes[0].len())
            .map(|d| {
                let mut column: Vec<f64> = shapes.iter().map(|s| s[d]).collect();
                column.sort_by(f64::total_cmp);
                column[column.len() / 2]
            })
            .collect();
        let score = |k: usize| {
            let along: f64 = shapes[k].iter().zip(&typical).map(|(x, y)| (x - y).abs()).sum::<f64>() / SHAPE_POINTS as f64;
            let c = &group[k];
            along + 0.5 * (distance(&c.frames[0], &mean[&a]) + distance(&c.frames[c.frames.len() - 1], &mean[&b]))
        };
        let scores: Vec<f64> = (0..group.len()).map(score).collect();
        let pick = (0..group.len()).min_by(|x, y| scores[*x].total_cmp(&scores[*y])).unwrap_or(0);
        let total = scores[pick];
        if let Some(chosen) = group.into_iter().nth(pick) {
            best.insert((a, b), (total, chosen));
        }
    }

    let mut index = Vec::new();
    let mut frames_bytes = Vec::new();
    for ((a, b), (_, c)) in &best {
        index.push(Entry { a: *a, b: *b, offset: (frames_bytes.len() / FRAME_BYTES) as u32, frames: c.frames.len() as u16, join: c.join as u16 });
        for f in &c.frames {
            frames_bytes.extend_from_slice(&f.lsf);
            frames_bytes.push(f.gain);
            frames_bytes.push(f.voiced);
        }
    }

    let mut ratios: HashMap<&str, Vec<f64>> = HashMap::new();
    for (take, found) in takes.iter().zip(&times) {
        let (Ok(found), Ok(planned)) = (found, plan_take(&g2p, &take.text)) else { continue };
        if found.len() == planned.segments.len() {
            for (t, s) in found.iter().zip(&planned.segments) {
                if s.1 > 0.0 {
                    ratios.entry(t.0.as_str()).or_default().push((t.2 - t.1) / s.1);
                }
            }
        }
    }
    let scales: Vec<u8> = phones
        .iter()
        .map(|p| match ratios.get_mut(p.as_str()) {
            Some(r) => {
                r.sort_by(f64::total_cmp);
                (r[r.len() / 2] * 64.0).round().clamp(32.0, 128.0) as u8
            }
            None => 64,
        })
        .collect();
    let file = voice_file(&phones, &scales, &pulse, &index, &frames_bytes);
    if let Some(dir) = std::path::Path::new(out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out_path, &file).map_err(|e| format!("{out_path}: {e}"))?;
    Ok(BuildReport { takes: takes.len() - failed, phones: phones.len(), joins: index.len(), frames: frames_bytes.len() / FRAME_BYTES, rom_bytes: file.len(), failed, single_example_joins })
}

fn voice_file(phones: &[String], scales: &[u8], pulse: &[i16], index: &[Entry], frames_bytes: &[u8]) -> Vec<u8> {
    let mut file = Vec::new();
    file.extend_from_slice(b"CVOX");
    file.push(phones.len() as u8);
    for (p, scale) in phones.iter().zip(scales) {
        file.push(p.len() as u8);
        file.extend_from_slice(p.as_bytes());
        file.push(*scale);
    }
    for k in 0..=256 {
        let c = (PI * k as f64 / 256.0).cos() * 32767.0;
        file.extend_from_slice(&(c.round() as i16).to_le_bytes());
    }
    for v in pulse {
        file.extend_from_slice(&v.to_le_bytes());
    }
    file.extend_from_slice(&(index.len() as u16).to_le_bytes());
    for e in index {
        file.push(e.a);
        file.push(e.b);
        file.extend_from_slice(&e.offset.to_le_bytes());
        file.extend_from_slice(&e.frames.to_le_bytes());
        file.extend_from_slice(&e.join.to_le_bytes());
    }
    file.extend_from_slice(frames_bytes);
    file
}

// ---- loading and speaking

pub struct LoadedVoice {
    pub pulse: Vec<i16>,
    /// how long this speaker holds each phone against the planner, Q6
    pub scales: Vec<u8>,
    pub joins: HashSet<(u8, u8)>,
    pub voice: Voice,
    pub phones: HashMap<String, u8>,
    pub rom_bytes: usize,
}

impl LoadedVoice {
    fn scale(&self, phone: u8) -> f64 {
        self.scales[phone as usize] as f64 / 64.0
    }
}

pub fn load(path: &str) -> Result<LoadedVoice, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 5 || &b[..4] != b"CVOX" {
        return Err(format!("{path} isn't a voice file"));
    }
    let bad = || format!("{path} is damaged");
    let mut at = 4;
    let n = *b.get(at).ok_or_else(bad)? as usize;
    at += 1;
    let mut phones = HashMap::new();
    let mut scales = Vec::with_capacity(n);
    for i in 0..n {
        let len = *b.get(at).ok_or_else(bad)? as usize;
        at += 1;
        let name = std::str::from_utf8(b.get(at..at + len).ok_or_else(bad)?).map_err(|_| bad())?;
        phones.insert(name.to_string(), i as u8);
        at += len;
        scales.push(*b.get(at).ok_or_else(bad)?);
        at += 1;
    }
    let rom_bytes = b.len();
    let pulse_at = at + 257 * 2;
    let pulse = b.get(pulse_at..pulse_at + PULSE_LEN * 2).ok_or_else(bad)?.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let index_at = pulse_at + PULSE_LEN * 2;
    let head = b.get(index_at..index_at + 2).ok_or_else(bad)?;
    let join_count = u16::from_le_bytes([head[0], head[1]]) as usize;
    let table = b.get(index_at + 2..index_at + 2 + join_count * 10).ok_or_else(bad)?;
    let joins = table.chunks(10).map(|e| (e[0], e[1])).collect();
    let voice = Voice::from_bytes(b).ok_or_else(bad)?;
    Ok(LoadedVoice { pulse, scales, joins, voice, phones, rom_bytes })
}

/// A character: all the tweakable settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Character {
    pub name: String,
    /// typical pitch (Hz)
    pub pitch: f64,
    /// how much pitch moves on stressed syllables (0..2)
    pub range: f64,
    /// speaking rate (1 is normal)
    pub rate: f64,
    /// vocal tract: above 1 smaller, below 1 larger
    pub size: f64,
    pub breath: f64,
    pub bright: f64,
    pub rough: f64,
}

impl Character {
    pub fn presets() -> Vec<Character> {
        let base = Character { name: "You".to_string(), pitch: 122.0, range: 1.0, rate: 1.0, size: 1.0, breath: 0.0, bright: 0.0, rough: 0.05 };
        vec![
            base.clone(),
            Character { name: "Giant".to_string(), pitch: 72.0, size: 0.84, rate: 0.85, range: 0.7, rough: 0.2, ..base.clone() },
            Character { name: "Goblin".to_string(), pitch: 240.0, size: 1.22, rate: 1.2, range: 1.5, bright: 0.3, ..base.clone() },
            Character { name: "Ghost".to_string(), pitch: 150.0, size: 1.06, rate: 0.9, breath: 0.75, range: 0.6, ..base },
        ]
    }

    fn params(&self) -> Params {
        Params { size: (self.size * 4096.0) as i32, breath: (self.breath * 4096.0) as i32, bright: (self.bright * 32767.0) as i32, rough: (self.rough * 4096.0) as i32 }
    }
}

pub struct Spoken {
    pub samples: Vec<i16>,
    pub stats: Stats,
    pub events: usize,
    pub words: Vec<SpokenWord>,
}

/// A word as written, and when it's said, in seconds from the start of the audio.
#[derive(Clone, Debug)]
pub struct SpokenWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A word or stretch the listener marked for later review.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub sentence: String,
    pub words: String,
    pub note: String,
    pub character: String,
    pub start: f64,
    pub end: f64,
    pub when: u64,
}

/// Each spoken word: the front end's text for it, and its first and last event.
type WordMarks = Vec<(String, usize, usize)>;

/// The front end lower-cases and spells out what it says; show the word as it was typed when it can be found.
fn display_words(text: &str, said: &[String]) -> Vec<String> {
    let tokens: Vec<&str> = text.split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}' || c == '-')).filter(|t| !t.is_empty()).collect();
    let mut at = 0;
    said.iter()
        .map(|w| match tokens[at..].iter().position(|t| t.replace('\u{2019}', "'").eq_ignore_ascii_case(w)) {
            Some(k) => {
                let token = tokens[at + k];
                at += k + 1;
                token.to_string()
            }
            None => w.clone(),
        })
        .collect()
}

/// Text to the synth's events: British phones, the planner's durations, a simple pitch contour.
pub fn events_for(g2p: &EnglishG2P, voice: &LoadedVoice, text: &str, c: &Character) -> Result<(Vec<Event>, WordMarks), String> {
    let phrases = g2p.to_phrases(text);
    if phrases.is_empty() {
        return Err("nothing to say".to_string());
    }
    let phrases = crate::british::southern_british(g2p, &phrases, "male", 112.0, false)?;
    let planned = plan(&inputs(&phrases), "male", 112.0);
    let spans = seg_spans(&phrases, &planned.segments)?;
    let labels = phone_labels(&phrases, &planned, &spans);
    let phone_id = |ipa: &str| -> Option<u8> { voice.phones.get(ipa).copied() };
    let silence = phone_id(SILENCE).ok_or("the voice has no silence")?;
    let schwa = phone_id("ə").unwrap_or(silence);
    let to_samples = |s: f64| ((s / c.rate) * FS_OUT).round().max(160.0) as u32;
    let hz = |f: f64| (f.clamp(40.0, 600.0) * 16.0) as u32;
    let mut events = vec![Event { phone: silence, samples: 3200, f0: hz(c.pitch) }];
    let said: Vec<String> = phrases.iter().flat_map(|p| p.words.iter().cloned()).collect();
    let mut marks: Vec<Option<(usize, usize)>> = vec![None; said.len()];
    for (pi, phrase) in phrases.iter().enumerate() {
        let in_phrase: Vec<_> = spans.iter().filter(|s| s.phrase == pi).collect();
        let phone_count: usize = in_phrase.iter().map(|s| s.count).sum();
        let last_vowel = in_phrase.iter().flat_map(|s| s.first..s.first + s.count).filter(|&k| planned.segments[k].kind == "vowel").max();
        let mut done = 0usize;
        for span in &in_phrase {
            for k in span.first..span.first + span.count {
                let s = &planned.segments[k];
                let progress = done as f64 / phone_count.max(1) as f64;
                let mut f = c.pitch * (1.08 - 0.2 * progress);
                if s.kind == "vowel" && s.stress.unwrap_or(0) == 1 {
                    f *= 1.0 + 0.18 * c.range;
                }
                if Some(k) == last_vowel {
                    match phrase.intonation {
                        Intonation::Question => f *= 1.0 + 0.3 * c.range,
                        Intonation::Statement => f *= 1.0 - 0.15 * c.range,
                        Intonation::Flat => {}
                    }
                }
                let phone = phone_id(&labels[k]).or_else(|| phone_id(&s.ipa)).unwrap_or(schwa);
                // the speaker's median shortens every example of a vowel, but "do" and "you" still need saying
                let floor = match labels[k].as_str() {
                    "ə" => 0.055,
                    "i" | "ɪ" | "ɛ" | "æ" | "ɑ" | "ɒ" | "ɔ" | "ʊ" | "u" | "ʌ" | "ɐ" => 0.08,
                    _ => 0.0,
                };
                if let Some(mark) = marks.get_mut(span.word) {
                    let first = mark.map_or(events.len(), |m| m.0);
                    *mark = Some((first, events.len()));
                }
                events.push(Event { phone, samples: to_samples(((s.t1 - s.t0) * voice.scale(phone)).max(floor)), f0: hz(f) });
                done += 1;
            }
        }
        events.push(Event { phone: silence, samples: to_samples(phrase.pause.max(0.1)), f0: hz(c.pitch * 0.9) });
    }
    let marks = said.into_iter().zip(marks).filter_map(|(w, m)| m.map(|(a, b)| (w, a, b))).collect();
    Ok((events, marks))
}

/// What the synth will do for a piece of text, sound by sound, to find out why a word sounds wrong.
pub fn explain(g2p: &EnglishG2P, voice: &LoadedVoice, text: &str, c: &Character) -> Result<String, String> {
    let (events, marks) = events_for(g2p, voice, text, c)?;
    let names: HashMap<u8, &str> = voice.phones.iter().map(|(n, i)| (*i, n.as_str())).collect();
    let mut out = String::new();
    for (word, first, last) in &marks {
        out.push_str(word);
        out.push_str(":");
        for k in *first..=*last {
            let e = &events[k];
            out.push_str(&format!(" {}({}ms)", names.get(&e.phone).copied().unwrap_or("?"), e.samples / 16));
            if k + 1 <= *last && !voice.joins.contains(&(e.phone, events[k + 1].phone)) {
                out.push_str(" [no join to next: borrowed]");
            }
        }
        out.push('\n');
    }
    Ok(out)
}

pub fn speak(g2p: &EnglishG2P, voice: &LoadedVoice, text: &str, c: &Character) -> Result<Spoken, String> {
    let (events, marks) = events_for(g2p, voice, text, c)?;
    let mut samples = vec![0i16; n64::output_len(&events)];
    let stats = n64::render(&voice.voice, &events, &c.params(), &mut samples);
    // the audio starts halfway through the first event
    let head = (events[0].samples / 2) as f64;
    let mut starts = Vec::with_capacity(events.len() + 1);
    let mut sum = 0.0;
    for e in &events {
        starts.push(sum);
        sum += e.samples as f64;
    }
    starts.push(sum);
    let said: Vec<String> = marks.iter().map(|m| m.0.clone()).collect();
    let words = marks.iter().zip(display_words(text, &said)).map(|((_, a, b), shown)| SpokenWord { text: shown, start: (starts[*a] - head) / FS_OUT, end: (starts[*b + 1] - head) / FS_OUT }).collect();
    Ok(Spoken { samples, stats, events: events.len(), words })
}

/// Rough share of the console's 93.75 MHz CPU to speak in real time: my estimate, not a measurement.
pub fn cpu_estimate_percent(stats: &Stats) -> f64 {
    if stats.samples == 0 {
        return 0.0;
    }
    // about 8 cycles per filter tap with its load and accumulate, 40 for the source per sample,
    // and about 2500 per update to rebuild the filter from its line spectral frequencies
    let cycles = stats.samples as f64 * (8.0 * ORDER as f64 + 40.0) + stats.filter_updates as f64 * 2500.0;
    let seconds = stats.samples as f64 / FS_OUT;
    100.0 * cycles / seconds / 93.75e6
}

pub fn to_float_at_fs(x: &[i16]) -> Vec<f32> {
    let f: Vec<f32> = x.iter().map(|&v| v as f32 / 32768.0).collect();
    resample(&f, FS_OUT, FS)
}

// ---- choosing what to read next

/// Words the speaker reported sounding wrong: their joins count double.
const FLAGGED: &str = "daisies please she'll do you for leaves passage seen pegs empty the air down";

/// Greedy choice of book sentences for the joins with the fewest recorded examples. Takes sentences
/// only before `CHAPTER IX`, and never one listed in `exclude`, so the rest of the book stays a fair test.
pub fn select_targeted(book: &str, recorded: &str, exclude: &str, out: &str, count: usize) -> Result<(), String> {
    const ENOUGH: usize = 3;
    type Join = (String, String);
    let g2p = pipeline::load_g2p()?;
    let joins_of = |text: &str| -> Option<(HashSet<Join>, Vec<Join>, usize)> {
        let p = plan_take(&g2p, text).ok()?;
        let all: Vec<Join> = p.segments.windows(2).filter(|w| w[0].2 == w[1].2).map(|w| (w[0].0.clone(), w[1].0.clone())).collect();
        Some((all.iter().cloned().collect(), all, p.segments.len()))
    };
    let mut have: HashMap<Join, usize> = HashMap::new();
    for line in std::fs::read_to_string(recorded).map_err(|e| format!("{recorded}: {e}"))?.lines() {
        let Some((_, text)) = line.split_once('\t') else { continue };
        if let Some((_, all, _)) = joins_of(text) {
            for j in all {
                *have.entry(j).or_default() += 1;
            }
        }
    }
    let excluded: HashSet<String> = std::fs::read_to_string(exclude).map_err(|e| format!("{exclude}: {e}"))?.lines().map(|l| l.trim().to_string()).collect();
    let flagged: HashSet<Join> = FLAGGED.split(' ').filter_map(|w| joins_of(w)).flat_map(|(set, _, _)| set).collect();
    let text = std::fs::read_to_string(book).map_err(|e| format!("{book}: {e}"))?;
    let (mut pool, mut frequency) = (Vec::new(), HashMap::<Join, f64>::new());
    for line in text.lines().take_while(|l| !l.starts_with("CHAPTER IX")) {
        let words = line.split_whitespace().count();
        let Some((set, all, phones)) = joins_of(line) else { continue };
        for j in all {
            *frequency.entry(j).or_default() += 1.0;
        }
        if (5..=14).contains(&words) && !line.starts_with("CHAPTER") && !line.chars().any(|c| c.is_ascii_digit()) && !excluded.contains(line.trim()) {
            pool.push((line.to_string(), set, phones));
        }
    }
    let short = |have: &HashMap<Join, usize>| frequency.keys().filter(|j| have.get(*j).copied().unwrap_or(0) < ENOUGH).count();
    let before = short(&have);
    let mut chosen = Vec::new();
    while chosen.len() < count && !pool.is_empty() {
        let value = |p: &(String, HashSet<Join>, usize)| {
            let sum: f64 = p.1.iter().filter(|j| have.get(*j).copied().unwrap_or(0) < ENOUGH).map(|j| frequency.get(j).copied().unwrap_or(1.0).ln_1p() * if flagged.contains(j) { 2.0 } else { 1.0 } / (1 + have.get(j).copied().unwrap_or(0)) as f64).sum();
            sum / p.2 as f64
        };
        let best = (0..pool.len()).max_by(|a, b| value(&pool[*a]).total_cmp(&value(&pool[*b]))).ok_or("empty pool")?;
        let picked = pool.swap_remove(best);
        for j in &picked.1 {
            *have.entry(j.clone()).or_default() += 1;
        }
        chosen.push(picked.0);
    }
    std::fs::write(out, chosen.join("\n") + "\n").map_err(|e| format!("{out}: {e}"))?;
    println!("chose {} sentences; joins in chapters I-VIII with under {ENOUGH} examples: {before} before, {} after, of {}", chosen.len(), short(&have), frequency.len());
    Ok(())
}

// ---- finding which stage loses the words

fn filled_pitch(f0: &[f64]) -> Vec<f64> {
    let mut last = f0.iter().copied().find(|f| *f > 0.0).unwrap_or(110.0);
    f0.iter()
        .map(|&f| {
            if f > 0.0 {
                last = f;
            }
            last
        })
        .collect()
}

fn q4(f: f64) -> u32 {
    (f.clamp(40.0, 600.0) * 16.0) as u32
}

/// A take's own frames, in order, through the synth: only the synth and the frame format can lose anything.
fn copy_synth(an: &Analysis, pulse: &[i16], params: &Params) -> Result<Vec<i16>, String> {
    const CHUNK: usize = 6; // 60 ms: a phone id per chunk, so pitch updates that often
    let n = (an.frames.len() / CHUNK).min(255);
    if n < 3 {
        return Err("take too short".to_string());
    }
    let pitch = filled_pitch(&an.f0);
    let names = vec![String::new(); n];
    let index: Vec<Entry> = (0..n - 1).map(|i| Entry { a: i as u8, b: (i + 1) as u8, offset: (i * CHUNK + CHUNK / 2) as u32, frames: CHUNK as u16, join: (CHUNK / 2) as u16 }).collect();
    let mut bytes = Vec::with_capacity(an.frames.len() * FRAME_BYTES);
    for f in &an.frames {
        bytes.extend_from_slice(&f.lsf);
        bytes.push(f.gain);
        bytes.push(f.voiced);
    }
    let voice = Voice::from_bytes(voice_file(&names, &vec![64; names.len()], pulse, &index, &bytes)).ok_or("bad copy voice")?;
    let events: Vec<Event> = (0..n).map(|i| Event { phone: i as u8, samples: (CHUNK * HOP) as u32, f0: q4(pitch[i * CHUNK + CHUNK / 2]) }).collect();
    let mut out = vec![0i16; n64::output_len(&events)];
    n64::render(&voice, &events, params, &mut out);
    Ok(out)
}

/// The stored joins, but with the recording's own phone timing and pitch.
fn joins_with_own_prosody(loaded: &LoadedVoice, times: &[(String, f64, f64, usize)], an: &Analysis, params: &Params) -> Result<Vec<i16>, String> {
    let silence = *loaded.phones.get(SILENCE).ok_or("the voice has no silence")?;
    let schwa = loaded.phones.get("ə").copied().unwrap_or(silence);
    let pitch = filled_pitch(&an.f0);
    let at = |t: f64| pitch[((t / 0.01) as usize).min(pitch.len() - 1)];
    let mut events = vec![Event { phone: silence, samples: 3200, f0: q4(at(0.0)) }];
    for (i, (ipa, t0, t1, phrase)) in times.iter().enumerate() {
        if i > 0 && times[i - 1].3 != *phrase {
            events.push(Event { phone: silence, samples: (((t0 - times[i - 1].2) * FS_OUT) as u32).max(160), f0: q4(at(*t0)) });
        }
        events.push(Event { phone: loaded.phones.get(ipa).copied().unwrap_or(schwa), samples: (((t1 - t0) * FS_OUT) as u32).max(160), f0: q4(at(0.5 * (t0 + t1))) });
    }
    events.push(Event { phone: silence, samples: 3200, f0: q4(at(1e9)) });
    let mut out = vec![0i16; n64::output_len(&events)];
    n64::render(&loaded.voice, &events, params, &mut out);
    Ok(out)
}

/// A take rebuilt from joins cut out of itself, through the normal playback path, so nothing can
/// differ from `copy_synth` but the cutting and stitching. With `durations`, each sound takes our
/// planned length instead of its recorded one, so the pieces have to stretch.
fn own_cuts(an: &Analysis, times: &[(String, f64, f64, usize)], durations: Option<&[f64]>, pulse: &[i16], params: &Params) -> Result<Vec<i16>, String> {
    let mut items: Vec<Option<usize>> = vec![None];
    for i in 0..times.len() {
        if i > 0 && times[i - 1].3 != times[i].3 {
            items.push(None);
        }
        items.push(Some(i));
    }
    items.push(None);
    if items.len() > 255 {
        return Err("take too long for a one-off voice".to_string());
    }
    let count = an.frames.len();
    let length = count as f64 * 0.01;
    let frame_at = |t: f64| (((t - 0.0125) / 0.01).round() as isize).clamp(0, count as isize - 1) as usize;
    let mid = |i: usize| 0.5 * (times[i].1 + times[i].2);
    let pitch = filled_pitch(&an.f0);
    let pitch_at = |t: f64| q4(pitch[((t / 0.01) as usize).min(pitch.len() - 1)]);
    let mut index = Vec::with_capacity(items.len());
    for k in 0..items.len() - 1 {
        let (t0, join, t1) = match (items[k], items[k + 1]) {
            (None, Some(b)) => ((times[b].1 - 0.06).max(0.0), times[b].1, mid(b)),
            (Some(a), None) => (mid(a), times[a].2, (times[a].2 + 0.06).min(length)),
            (Some(a), Some(b)) => (mid(a), times[a].2, mid(b)),
            (None, None) => return Err("two silences in a row".to_string()),
        };
        let i0 = frame_at(t0).min(count.saturating_sub(3));
        let j = frame_at(join).max(i0 + 1);
        let i1 = frame_at(t1).max(j + 1).min(count);
        index.push(Entry { a: k as u8, b: (k + 1) as u8, offset: i0 as u32, frames: (i1 - i0) as u16, join: (j - i0) as u16 });
    }
    let mut bytes = Vec::with_capacity(count * FRAME_BYTES);
    for f in &an.frames {
        bytes.extend_from_slice(&f.lsf);
        bytes.push(f.gain);
        bytes.push(f.voiced);
    }
    let names = vec![String::new(); items.len()];
    let voice = Voice::from_bytes(voice_file(&names, &vec![64; names.len()], pulse, &index, &bytes)).ok_or("bad one-off voice")?;
    let mut events = Vec::with_capacity(items.len());
    for (k, item) in items.iter().enumerate() {
        let (samples, f0) = match *item {
            Some(i) => {
                let seconds = durations.map_or(times[i].2 - times[i].1, |d| d[i]);
                (((seconds * FS_OUT) as u32).max(160), pitch_at(mid(i)))
            }
            None if k == 0 || k + 1 == items.len() => (3200, pitch_at(if k == 0 { 0.0 } else { length })),
            None => {
                let (before, after) = (items[k - 1].ok_or("silence")?, items[k + 1].ok_or("silence")?);
                ((((times[after].1 - times[before].2) * FS_OUT) as u32).max(160), pitch_at(times[after].1))
            }
        };
        events.push(Event { phone: k as u8, samples, f0 });
    }
    let mut out = vec![0i16; n64::output_len(&events)];
    n64::render(&voice, &events, params, &mut out);
    Ok(out)
}

/// Writes the same takes several ways into `<out>/<stage>/say<i>.wav`, with `tests.txt` beside them.
pub fn ladder(takes: &[Take], voice_path: &str, out_dir: &str) -> Result<(), String> {
    let g2p = pipeline::load_g2p()?;
    let loaded = load(voice_path)?;
    let character = Character::presets().remove(0);
    let params = character.params();
    let work = format!("{out_dir}/work");
    std::fs::create_dir_all(&work).map_err(|e| format!("{work}: {e}"))?;
    let times = mfa_align(&g2p, takes, &work)?;
    let mut recordings = Vec::new();
    for t in takes {
        recordings.push(read_wav(&t.wav)?);
    }
    let located: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = takes.iter().zip(&times).map(|(t, found)| s.spawn(move || -> Result<(PhoneTimes, Analysis), String> {
            let found = found.clone()?;
            let mut an = analysis_of(t)?;
            mark_voiced(&mut an, &found);
            Ok((found, an))
        })).collect();
        handles.into_iter().map(|h| h.join().expect("analysis thread")).collect()
    });
    let stages = ["1-recording", "2-copy", "2a-own-cuts", "2b-own-cuts-our-timing", "3-joins", "4-tts"];
    let tests = takes.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join("\n") + "\n";
    for stage in stages {
        std::fs::create_dir_all(format!("{out_dir}/{stage}")).map_err(|e| e.to_string())?;
        std::fs::write(format!("{out_dir}/{stage}/tests.txt"), &tests).map_err(|e| e.to_string())?;
    }
    for (i, (take, found)) in takes.iter().zip(located).enumerate() {
        let (times, an) = found?;
        let wav = |stage: &str| format!("{out_dir}/{stage}/say{i}.wav");
        write_wav(&wav(stages[0]), &recordings[i])?;
        write_wav(&wav(stages[1]), &to_float_at_fs(&copy_synth(&an, &loaded.pulse, &params)?))?;
        write_wav(&wav(stages[2]), &to_float_at_fs(&own_cuts(&an, &times, None, &loaded.pulse, &params)?))?;
        let planned: Vec<f64> = plan_take(&g2p, &take.text)?.segments.iter().map(|s| s.1 * loaded.phones.get(&s.0).map_or(1.0, |&p| loaded.scale(p))).collect();
        write_wav(&wav(stages[3]), &to_float_at_fs(&own_cuts(&an, &times, Some(&planned), &loaded.pulse, &params)?))?;
        write_wav(&wav(stages[4]), &to_float_at_fs(&joins_with_own_prosody(&loaded, &times, &an, &params)?))?;
        write_wav(&wav(stages[5]), &to_float_at_fs(&speak(&g2p, &loaded, &take.text, &character)?.samples))?;
    }
    let _ = std::fs::remove_dir_all(&work);
    Ok(())
}

// ---- choosing what to read

/// Greedy selection from Arctic-format lines: most new joins per sound.
pub fn select_prompts(pool_path: &str, out_path: &str, count: usize) -> Result<(), String> {
    let g2p = pipeline::load_g2p()?;
    let text = std::fs::read_to_string(pool_path).map_err(|e| format!("{pool_path}: {e}"))?;
    let mut pool: Vec<(String, HashSet<(String, String)>, usize)> = Vec::new();
    for line in text.lines() {
        let (Some(a), Some(b)) = (line.find('"'), line.rfind('"')) else { continue };
        if b <= a {
            continue;
        }
        let sentence = line[a + 1..b].trim().to_string();
        let words = sentence.split_whitespace().count();
        if !(4..=12).contains(&words) || sentence.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        let phrases = g2p.to_phrases(&sentence);
        if phrases.is_empty() {
            continue;
        }
        let phrases = crate::british::southern_british(&g2p, &phrases, "male", 112.0, false)?;
        let p = plan(&inputs(&phrases), "male", 112.0);
        let phones: Vec<String> = p.segments.iter().map(|s| s.ipa.clone()).collect();
        let joins: HashSet<(String, String)> = phones.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect();
        pool.push((sentence, joins, phones.len()));
    }
    let total: HashSet<(String, String)> = pool.iter().flat_map(|p| p.1.iter().cloned()).collect();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut chosen = Vec::new();
    while chosen.len() < count && !pool.is_empty() {
        let (best, _) = pool.iter().enumerate().map(|(i, p)| (i, p.1.iter().filter(|j| !seen.contains(*j)).count() as f64 / p.2 as f64)).max_by(|a, b| a.1.total_cmp(&b.1)).ok_or("empty pool")?;
        let picked = pool.swap_remove(best);
        seen.extend(picked.1.iter().cloned());
        chosen.push(picked.0);
    }
    std::fs::write(out_path, chosen.join("\n") + "\n").map_err(|e| format!("{out_path}: {e}"))?;
    println!("chose {} sentences covering {} of {} joins seen in the pool", chosen.len(), seen.len(), total.len());
    Ok(())
}

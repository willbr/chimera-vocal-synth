//! Southern British pronunciation over the American dictionary's output, on the segment lists, so
//! the engine itself doesn't change. A spike's approximation, not a full accent model:
//!   - no /r/ unless a vowel follows (linking r kept); "year" → ɪə, "sure" → ʊə, "water" → ə
//!   - /t d/ stay stops instead of the American flap (no glottal stops yet)
//!   - LOT is ɒ, BATH words are ɑ, GOAT glides ə → ʊ
//!   - NURSE borrows ə: the engine has no non-rhotic ɜ target yet
//!   - GOOSE stays the engine's back /u/; fronting it is for the vowel fit

use crate::pipeline::{inputs, plan, seg_spans};
use chimera::phonetics::Segment;
use chimera::phonetics::english::{EnglishG2P, Phrase};

const VOWELS: [&str; 18] = ["ɑ", "æ", "ʌ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ɝ", "ɚ", "eɪ", "ɪ", "i", "oʊ", "ɔɪ", "ʊ", "u", "ɒ"];

/// The American /æ/ words that take the long "father" vowel in the south, and /ɑ/ words that keep it.
const LONG_A: &str = "ask asks asked asking after afternoon answer answers answered aunt aunts bath baths basket branch branches brass calf can't castle chance chances class classes command craft dance dances danced dancing demand draft example examples fast faster father france ghastly glass glasses grant graph grass half laugh laughs laughed laughing last lasts lasted lasting mask master masters nasty pass passes passed passing past path paths plant plants plaster rather sample samples shan't slant staff task tasks vast advance advanced disaster calm palm spa bra drama ma pa";

fn long_a(word: &str) -> bool {
    LONG_A.split(' ').any(|w| w == word)
}

fn is_vowel(ipa: &str) -> bool {
    VOWELS.contains(&ipa)
}

/// One output segment. `split` is a two-vowel glide: planned first as `seg.ipa` so it gets that
/// sound's duration, then sung as the two parts sharing it.
struct Out {
    seg: Segment,
    split: Option<(&'static str, &'static str)>,
    /// the "nurse" vowel: said as ə unless a fitted one exists
    nurse: bool,
    /// on top of the planner's length: "air" keeps the time its lost /r/ gave it
    stretch: f64,
}

fn rewrite(g2p: &EnglishG2P, phrase: &Phrase) -> Vec<Out> {
    // flapping lost whether a tap was /t/ or /d/; the dictionary's word entries still know
    let base: Vec<Segment> = phrase.words.iter().flat_map(|w| g2p.word(w, true)).collect();
    let same_shape = base.len() == phrase.segments.len();
    let segs = &phrase.segments;
    let mut out: Vec<Out> = Vec::with_capacity(segs.len() + 4);
    let mut word = 0;
    for (i, s) in segs.iter().enumerate() {
        let vowel_next = segs.get(i + 1).is_some_and(|n| is_vowel(&n.ipa));
        let coda_r_next = segs.get(i + 1).is_some_and(|n| n.ipa == "ɹ") && !segs.get(i + 2).is_some_and(|n| is_vowel(&n.ipa));
        let w = phrase.words.get(word).map_or("", String::as_str);
        let mut seg = s.clone();
        let mut split = None;
        let mut nurse = false;
        match s.ipa.as_str() {
            "ɹ" if !vowel_next && i > 0 && is_vowel(&segs[i - 1].ipa) => {
                // the /r/ goes; the vowel before it becomes what's left of it
                if let Some(prev) = out.last_mut() {
                    match prev.seg.ipa.as_str() {
                        "ɪ" | "i" => {
                            prev.seg.ipa = "ɪ".to_string();
                            prev.split = Some(("ɪ", "ə"));
                        }
                        "ʊ" | "u" => {
                            prev.seg.ipa = "ʊ".to_string();
                            prev.split = Some(("ʊ", "ə"));
                        }
                        "ɛ" => prev.stretch = 1.7,
                        _ => {}
                    }
                    if s.word_end == Some(true) {
                        prev.seg.word_end = Some(true);
                    }
                }
                if s.word_end == Some(true) {
                    word += 1;
                }
                continue;
            }
            "ɾ" if same_shape => seg.ipa = base[i].ipa.clone(),
            // these keep their vowel even unstressed: a bare schwa is heard as a stray "uh"
            "ɚ" | "ə" if w == "or" || w == "nor" => seg.ipa = "ɔ".to_string(),
            // the dictionary's American or weak forms that southern English doesn't use
            "ə" if w == "on" => seg.ipa = "ɒ".to_string(),
            "ə" if matches!(w, "would" | "could" | "should") => seg.ipa = "ʊ".to_string(),
            "ʌ" if matches!(w, "what" | "what's" | "because") => seg.ipa = "ɒ".to_string(),
            "ɚ" | "ɝ" => {
                if vowel_next {
                    // "water all", "hurry": the /r/ survives before a vowel
                    let mut r = Segment { ipa: "ɹ".to_string(), stress: Some(0), word_end: s.word_end, ..Default::default() };
                    seg.ipa = if s.ipa == "ɝ" { "ʌ" } else { "ə" }.to_string();
                    seg.word_end = None;
                    out.push(Out { seg, split: None, nurse: false, stretch: 1.0 });
                    r.dur = None;
                    out.push(Out { seg: r, split: None, nurse: false, stretch: 1.0 });
                    if s.word_end == Some(true) {
                        word += 1;
                    }
                    continue;
                }
                seg.ipa = "ə".to_string();
                nurse = s.ipa == "ɝ";
            }
            "oʊ" => split = Some(("ə", "ʊ")),
            // "the air" is "thee air"
            "ə" if w == "the" && s.word_end == Some(true) && vowel_next => seg.ipa = "i".to_string(),
            // "passage", "village": southern English has ɪ here
            // and "rabbit", "wanted", "horses"
            "ə" if s.stress != Some(1) && segs.get(i + 1).is_some_and(|n| n.word_end == Some(true) && (n.ipa == "dʒ" || (n.ipa == "t" && w.ends_with("it")) || (n.ipa == "d" && w.ends_with("ed")) || (n.ipa == "z" && w.ends_with("es")))) => seg.ipa = "ɪ".to_string(),
            "æ" if long_a(w) => seg.ipa = "ɑ".to_string(),
            "ɑ" if !coda_r_next && !long_a(w) => seg.ipa = "ɒ".to_string(),
            "ɔ" if segs.get(i + 1).is_some_and(|n| ["f", "θ", "s", "ŋ", "g"].contains(&n.ipa.as_str())) => seg.ipa = "ɒ".to_string(),
            _ => {}
        }
        if s.word_end == Some(true) {
            word += 1;
        }
        // NURSE is a long vowel; written as a schwa it was as short as one
        out.push(Out { seg, split, nurse, stretch: if nurse { 1.6 } else { 1.0 } });
    }
    out
}

/// The phrases as a southern British speaker would say them, every duration made explicit.
/// `fitted_nurse` says the unused ɐ will carry a fitted "nurse" shape (see `fit::apply`).
pub fn southern_british(g2p: &EnglishG2P, phrases: &[Phrase], body: &str, f0: f64, fitted_nurse: bool) -> Result<Vec<Phrase>, String> {
    let rewritten: Vec<Vec<Out>> = phrases.iter().map(|p| rewrite(g2p, p)).collect();
    let pass1: Vec<Phrase> = phrases.iter().zip(&rewritten).map(|(p, out)| Phrase { segments: out.iter().map(|o| o.seg.clone()).collect(), intonation: p.intonation, pause: p.pause, words: p.words.clone() }).collect();
    let planned = plan(&inputs(&pass1), body, f0);
    let spans = seg_spans(&pass1, &planned.segments)?;
    let mut finished: Vec<Phrase> = pass1;
    let mut parts: Vec<Vec<Segment>> = finished.iter().map(|p| Vec::with_capacity(p.segments.len() + 4)).collect();
    for span in &spans {
        let out = &rewritten[span.phrase][span.seg];
        let dur = planned.segments[span.first + span.count - 1].t1 - planned.segments[span.first].t0;
        let mut seg = out.seg.clone();
        match out.split {
            Some((a, b)) => {
                let first = Segment { ipa: a.to_string(), dur: Some(dur * 0.55), stress: seg.stress, ..Default::default() };
                seg.ipa = b.to_string();
                seg.dur = Some(dur * 0.45);
                parts[span.phrase].push(first);
                parts[span.phrase].push(seg);
            }
            None => {
                // only a voice with its own NURSE sound takes the longer length
                seg.dur = Some(dur * if out.nurse && !fitted_nurse { 1.0 } else { out.stretch });
                if out.nurse && fitted_nurse {
                    seg.ipa = "ɐ".to_string();
                }
                parts[span.phrase].push(seg);
            }
        }
    }
    for (p, segs) in finished.iter_mut().zip(parts) {
        p.segments = segs;
    }
    Ok(finished)
}

/// "wɔːtə" style: the phonemes the engine will say, words separated by spaces.
pub fn phones(phrases: &[Phrase]) -> String {
    phrases
        .iter()
        .map(|p| {
            let mut s = String::new();
            for seg in &p.segments {
                if seg.stress == Some(1) && (is_vowel(&seg.ipa) || seg.ipa == "ɐ") {
                    s.push('ˈ');
                }
                s.push_str(crate::fit::fitted_label(&seg.ipa));
                if seg.word_end == Some(true) {
                    s.push(' ');
                }
            }
            s.trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

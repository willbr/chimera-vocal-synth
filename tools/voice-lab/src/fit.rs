//! A mouth shape for each of the speaker's vowels: the search `chimera calibrate vowels` does, aimed
//! at formants measured from their recorded words instead of textbook means. The fitted shapes
//! replace the planner's vowel gestures in Voice Lab only; the engine and its tables don't change.

use crate::pipeline::{HeldVowel, WORDS, engine_vowel_table, repo_path};
use chimera::airway::frequency::{find_peaks, transfer_function};
use chimera::anatomy::human::{ArticKey, BODY_FEMALE, BODY_MALE, HumanArtic, HumanBody, NEUTRAL, rest_tip};
use chimera::phonetics::Plan;
use chimera::util::optimize::{NelderMeadOpts, nelder_mead};
use chimera::voice::human_airway::HumanAirway;
use serde::{Deserialize, Serialize};

/// The planner's vowel keys, in the order of the fit's parameters.
const KEYS: [ArticKey; 9] = [ArticKey::TbX, ArticKey::TbY, ArticKey::Jaw, ArticKey::LipAp, ArticKey::LipProt, ArticKey::LipWidth, ArticKey::Root, ArticKey::Larynx, ArticKey::TtCurl];
const TABLE_KEYS: [&str; 9] = ["tbX", "tbY", "jaw", "lipAp", "lipProt", "lipWidth", "root", "larynx", "ttCurl"];
/// As `chimera calibrate`: tbX tbY jaw lipAp lipProt lipWidth root larynx curl.
const LO: [f64; 9] = [1.2, 5.0, 0.05, 0.12, 0.0, 0.0, -0.9, -1.3, 0.0];
const HI: [f64; 9] = [4.6, 8.8, 2.3, 2.0, 1.4, 1.0, 0.9, 1.3, 1.0];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitVowel {
    pub ipa: String,
    pub word: String,
    pub posture: [f64; 9],
    pub target: [f64; 3],
    pub achieved: [f64; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fit {
    pub body: String,
    pub vowels: Vec<FitVowel>,
}

impl Fit {
    pub fn get(&self, ipa: &str) -> Option<&FitVowel> {
        self.vowels.iter().find(|v| v.ipa == ipa)
    }

    /// Mean |achieved − target| / target (%) for F1, F2, F3.
    pub fn error_pct(&self) -> [f64; 3] {
        let mut sum = [0.0; 3];
        for v in &self.vowels {
            for k in 0..3 {
                sum[k] += 100.0 * (v.achieved[k] - v.target[k]).abs() / v.target[k];
            }
        }
        sum.map(|s| s / self.vowels.len().max(1) as f64)
    }
}

fn to_artic(p: &[f64]) -> HumanArtic {
    let tip = rest_tip(p[2], p[0], p[1]);
    HumanArtic { tb_x: p[0], tb_y: p[1], jaw: p[2], lip_ap: p[3], lip_prot: p[4], lip_width: p[5], root: p[6], larynx: p[7], tt_x: tip.x - 0.4 * p[8], tt_y: tip.y + 2.0 * p[8], tt_curl: p[8], ..NEUTRAL }
}

fn formants(air: &mut HumanAirway, freqs: &[f64], a: &HumanArtic) -> (Vec<f64>, f64) {
    air.apply(a, 0.0);
    let el = air.net.linear_elements();
    let h = transfer_function(&el, air.pharynx.first_node(), freqs, &[air.glottis_port]);
    let peaks: Vec<f64> = find_peaks(freqs, &h.mag, 150.0).iter().map(|p| p.freq).collect();
    let min_area = air.ph_areas.iter().chain(air.or_areas.iter()).copied().fold(f64::INFINITY, f64::min);
    (peaks, min_area)
}

fn fit_one(body: HumanBody, start: [f64; 9], target: [f64; 3]) -> ([f64; 9], [f64; 3]) {
    let n = 700;
    let freqs: Vec<f64> = (0..n).map(|i| 120.0 * (4600.0f64 / 120.0).powf(i as f64 / (n - 1) as f64)).collect();
    let mut air = HumanAirway::new(96000.0, body);
    let min_ok = 0.22 * body.scale * body.scale;
    let mut cost = |p: &[f64]| -> f64 {
        let (f, min_area) = formants(&mut air, &freqs, &to_artic(p));
        if f.len() < 3 {
            return 50.0;
        }
        let w = [1.0, 1.0, 0.45];
        let mut c: f64 = (0..3).map(|i| w[i] * (f[i] / target[i]).ln().powi(2)).sum();
        if min_area < min_ok {
            c += 0.4 * ((min_ok - min_area) / min_ok).powi(2);
        }
        // Many mouth shapes give the same formants: stay near the engine's shape for this vowel,
        // so transitions stay short and the tongue doesn't wander. None of these are rhotic.
        c += 0.01 * (0..9).map(|i| ((p[i] - start[i]) / (HI[i] - LO[i])).powi(2)).sum::<f64>();
        c + 0.5 * p[8] * p[8]
    };
    let opts = |scale: f64, max_iter: usize| NelderMeadOpts {
        lo: Some(LO.to_vec()),
        hi: Some(HI.to_vec()),
        step: Some([0.4, 0.4, 0.3, 0.25, 0.3, 0.25, 0.3, 0.3, 0.2].iter().map(|s| s * scale).collect()),
        max_iter: Some(max_iter),
        tol: None,
    };
    let mut best = nelder_mead(&mut cost, &start, opts(1.0, 1800));
    for _ in 0..2 {
        let next = nelder_mead(&mut cost, &best.x, opts(0.5, 1000));
        if next.fx < best.fx {
            best = next;
        }
    }
    let (f, _) = formants(&mut air, &freqs, &to_artic(&best.x));
    let mut posture = [0.0; 9];
    posture.copy_from_slice(&best.x);
    (posture, [f.first().copied().unwrap_or(0.0), f.get(1).copied().unwrap_or(0.0), f.get(2).copied().unwrap_or(0.0)])
}

fn fit_path() -> String {
    repo_path("out/voice-lab/fit.json")
}

/// Fit every recorded word's vowel and save the result. `body` is "male" or "female".
pub fn fit_recordings(body: &str) -> Result<Fit, String> {
    let shape = if body == "female" { BODY_FEMALE } else { BODY_MALE };
    let table = engine_vowel_table();
    let mut jobs: Vec<(&str, &str, [f64; 9], [f64; 3])> = Vec::new();
    for (id, ipa, word, _) in WORDS {
        let path = repo_path(&format!("out/voice-lab/words/{id}.json"));
        let Some(held) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<HeldVowel>(&t).ok()) else { continue };
        // the engine has no non-rhotic "nurse": start that one from its schwa
        let source = if ipa == "ɜ" { "ə" } else { ipa };
        let artic = table.get(body).and_then(|b| b.get(source)).and_then(|v| v.get("artic")).ok_or_else(|| format!("no engine shape for {source}"))?;
        let mut start = [0.0; 9];
        for k in 0..9 {
            start[k] = artic.get(TABLE_KEYS[k]).and_then(|x| x.as_f64()).unwrap_or(0.0).clamp(LO[k], HI[k]);
        }
        jobs.push((ipa, word, start, held.formants));
    }
    if jobs.is_empty() {
        return Err("no words recorded yet".to_string());
    }
    let vowels: Vec<FitVowel> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(ipa, word, start, target)| {
                s.spawn(move || {
                    let (posture, achieved) = fit_one(shape, start, target);
                    FitVowel { ipa: ipa.to_string(), word: word.to_string(), posture, target, achieved }
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let fit = Fit { body: body.to_string(), vowels };
    let json = serde_json::to_string_pretty(&fit).map_err(|e| e.to_string())?;
    std::fs::write(fit_path(), json).map_err(|e| format!("{}: {e}", fit_path()))?;
    Ok(fit)
}

pub fn load() -> Option<Fit> {
    std::fs::read_to_string(fit_path()).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// The vowel a gesture label stands for: the plan borrows the unused ɐ to carry fitted "nurse".
pub fn fitted_label(label: &str) -> &str {
    if label == "ɐ" { "ɜ" } else { label }
}

/// Swap each vowel gesture's mouth shape for the fitted one, leaving its timing alone.
pub fn apply(plan: &mut Plan, fit: &Fit) {
    for g in plan.gestures.iter_mut() {
        let Some(label) = g.label else { continue };
        let Some(v) = fit.get(fitted_label(label.as_str())) else { continue };
        for (k, key) in KEYS.iter().enumerate() {
            if g.targets.get(*key).is_some() {
                g.targets.set(*key, v.posture[k]);
            }
        }
    }
}

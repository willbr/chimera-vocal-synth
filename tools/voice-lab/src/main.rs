//! Voice Lab: record a line, hear it next to the engine's version, and see how the two differ.
//! A throwaway spike. Everything it writes goes under `out/voice-lab/`, which git ignores.
//!
//!   voice-lab                                       the window
//!   voice-lab --analyze in.wav "text" outdir [body] [british|american]   the same, no window
//!   voice-lab --check-audio                         open each device without capturing anything

// A window with no console beside it. The command-line modes attach to the terminal they ran from.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod audio;
mod british;
mod fit;
mod n64;
mod pipeline;
mod voice;

use audio::Recording;
use eframe::egui;
use pipeline::{FS, HeldVowel, Report, WORDS, repo_path};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

const DEFAULT_LINES: [&str; 3] = [
    "She had your dark suit in greasy wash water all year.",
    "Don't ask me to carry an oily rag like that.",
    "The rainbow is a division of white light into many beautiful colors.",
];
const BODIES: [&str; 3] = ["auto", "male", "female"];
const ACCENTS: [(&str, &str); 2] = [("british", "Southern British"), ("american", "American (engine default)")];

#[derive(PartialEq)]
enum Page {
    Lines,
    Vowels,
    Read,
    Speak,
}

fn voice_path() -> String {
    format!("{}/voice.bin", lab_dir())
}

fn notes_path() -> String {
    format!("{}/notes.jsonl", lab_dir())
}

/// The next sentence to show, passing over any marked in `skip`.
fn next_prompt(skip: &[bool], from: usize, forward: bool) -> Option<usize> {
    if forward { (from + 1..skip.len()).find(|&i| !skip[i]) } else { (0..from).rev().find(|&i| !skip[i]) }
}

/// For each word of a spoken sentence, the open notes on it, by index into `notes`. A note is
/// found by its words, then by the closest time, because timing moves as voices change.
fn notes_on_words(notes: &[voice::Note], sentence: &str, words: &[voice::SpokenWord]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); words.len()];
    for (index, n) in notes.iter().enumerate().filter(|(_, n)| !n.fixed && n.sentence == sentence) {
        let wanted: Vec<&str> = n.words.split(' ').collect();
        let starts = 0..words.len().saturating_sub(wanted.len() - 1);
        let matches = starts.filter(|&i| wanted.iter().enumerate().all(|(k, w)| words[i + k].text == *w));
        let distance = |i: usize| (words[i].start - n.start).abs();
        if let Some(i) = matches.min_by(|&a, &b| distance(a).total_cmp(&distance(b))) {
            for k in 0..wanted.len() {
                out[i + k].push(index);
            }
        }
    }
    out
}

fn characters_path() -> String {
    format!("{}/characters.json", lab_dir())
}

fn prompt_wav(i: usize) -> String {
    format!("{}/read/{i:03}.wav", lab_dir())
}

fn vowels_dir() -> String {
    format!("{}/words", lab_dir())
}

fn vowel_file(id: &str, ext: &str) -> String {
    format!("{}/{id}.{ext}", vowels_dir())
}

fn lab_dir() -> String {
    repo_path("out/voice-lab")
}

fn lines_file() -> String {
    format!("{}/lines.txt", lab_dir())
}

fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    out.trim_end_matches('-').to_string()
}

fn line_dir(text: &str) -> String {
    format!("{}/{}", lab_dir(), slug(text))
}

enum Event {
    Status(String),
    Done(Result<Report, String>),
    Fitted(Result<fit::Fit, String>),
    Built(Result<voice::BuildReport, String>),
}

struct App {
    lines: Vec<String>,
    selected: usize,
    inputs: Vec<String>,
    input: Option<String>,
    body: usize,
    accent: usize,
    recording: Option<Recording>,
    player: Option<cpal::Stream>,
    status: String,
    busy: Option<Receiver<Event>>,
    report: Option<Report>,
    page: Page,
    held: Vec<Option<HeldVowel>>,
    recording_vowel: Option<usize>,
    vowel_table: serde_json::Value,
    fit: Option<fit::Fit>,
    prompts: Vec<String>,
    prompt: usize,
    prompt_done: Vec<bool>,
    /// recorded before this launch: the takes of this session stay reachable, to redo a mistake
    prompt_old: Vec<bool>,
    skip_old: bool,
    recording_prompt: Option<usize>,
    g2p: Option<chimera::phonetics::english::EnglishG2P>,
    loaded: Option<voice::LoadedVoice>,
    characters: Vec<voice::Character>,
    character: usize,
    say_text: String,
    speak_info: String,
    book: Vec<&'static str>,
    book_pos: usize,
    read_on: bool,
    speak_until: Option<std::time::Instant>,
    spoken_words: Vec<voice::SpokenWord>,
    spoken_audio: Vec<f32>,
    play_started: Option<std::time::Instant>,
    play_offset: f64,
    play_end: f64,
    selection: Option<(usize, usize)>,
    note_text: String,
    /// The notes open for editing, by index into `notes`, each with its text so far.
    note_edits: Vec<(usize, String)>,
    notes: Vec<voice::Note>,
    spoken_text: String,
    show_fixed: bool,
}

/// Alice's Adventures in Wonderland (1865, public domain), one sentence per line.
const BOOK: &str = include_str!("../books/alice.txt");

impl App {
    fn new() -> App {
        let lines = std::fs::read_to_string(lines_file())
            .ok()
            .map(|t| t.lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect::<Vec<_>>())
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| DEFAULT_LINES.iter().map(|s| s.to_string()).collect());
        let mut app = App {
            lines,
            selected: 0,
            inputs: audio::input_names(),
            input: audio::default_input_name(),
            body: 0,
            accent: 0,
            recording: None,
            player: None,
            status: "Pick a line, press Record, read it, press Stop.".to_string(),
            busy: None,
            report: None,
            page: Page::Lines,
            held: WORDS.iter().map(|(id, ..)| std::fs::read_to_string(vowel_file(id, "json")).ok().and_then(|t| serde_json::from_str(&t).ok())).collect(),
            recording_vowel: None,
            vowel_table: pipeline::engine_vowel_table(),
            fit: fit::load(),
            prompts: Vec::new(),
            prompt: 0,
            prompt_done: Vec::new(),
            prompt_old: Vec::new(),
            skip_old: true,
            recording_prompt: None,
            g2p: None,
            loaded: voice::load(&voice_path()).ok(),
            characters: std::fs::read_to_string(characters_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).filter(|c: &Vec<voice::Character>| !c.is_empty()).unwrap_or_else(voice::Character::presets),
            character: 0,
            say_text: "Hello there. I have been waiting for you all morning.".to_string(),
            speak_info: String::new(),
            book: BOOK.lines().collect(),
            book_pos: 0,
            read_on: false,
            speak_until: None,
            spoken_words: Vec::new(),
            spoken_audio: Vec::new(),
            play_started: None,
            play_offset: 0.0,
            play_end: 0.0,
            selection: None,
            note_text: String::new(),
            note_edits: Vec::new(),
            notes: std::fs::read_to_string(notes_path()).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default(),
            spoken_text: String::new(),
            show_fixed: false,
        };
        app.prompts = std::fs::read_to_string(repo_path("out/voice-lab/prompts.txt")).map(|t| t.lines().map(str::to_string).filter(|l| !l.trim().is_empty()).collect()).unwrap_or_default();
        app.prompts.extend(include_str!("../books/alice-prompts.txt").lines().map(str::to_string));
        app.prompt_done = (0..app.prompts.len()).map(|i| Path::new(&prompt_wav(i)).exists()).collect();
        app.prompt_old = app.prompt_done.clone();
        app.prompt = app.prompt_done.iter().position(|d| !d).unwrap_or(0);
        app.load_report();
        app
    }

    fn text(&self) -> &str {
        &self.lines[self.selected]
    }

    fn file(&self, name: &str) -> String {
        format!("{}/{name}", line_dir(self.text()))
    }

    fn has(&self, name: &str) -> bool {
        Path::new(&self.file(name)).exists()
    }

    fn load_report(&mut self) {
        self.report = std::fs::read_to_string(self.file("report.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).filter(|r: &Report| r.text == self.text());
    }

    fn save_lines(&self) {
        let _ = std::fs::create_dir_all(lab_dir());
        let _ = std::fs::write(lines_file(), self.lines.join("\n"));
    }

    fn start_recording(&mut self) {
        self.player = None;
        match Recording::start(self.input.as_deref()) {
            Ok(r) => {
                self.recording = Some(r);
                self.status = "Recording. Read the line, then press Stop.".to_string();
            }
            Err(e) => self.status = format!("Couldn't open the microphone: {e}"),
        }
    }

    fn stop_recording(&mut self) {
        let Some(r) = self.recording.take() else { return };
        let x = r.stop();
        let peak = x.iter().fold(0f32, |p, v| p.max(v.abs()));
        let path = self.file("mine.wav");
        match pipeline::write_wav(&path, &x) {
            Ok(()) => {
                self.report = None;
                let secs = x.len() as f64 / FS;
                self.status = if peak >= 0.99 {
                    format!("Saved {secs:.1} s, but it clipped. Move back from the mic and record again.")
                } else if peak < 0.03 {
                    format!("Saved {secs:.1} s, but it's very quiet (peak {peak:.3}). Check the microphone choice.")
                } else {
                    format!("Saved {secs:.1} s (peak {peak:.2}). Play it back or press Compare.")
                };
            }
            Err(e) => self.status = format!("Couldn't save: {e}"),
        }
    }

    fn play(&mut self, name: &str) {
        self.play_path(&self.file(name));
    }

    fn play_path(&mut self, path: &str) {
        self.player = None;
        match pipeline::read_wav(path).and_then(|x| audio::play(&x)) {
            Ok(s) => self.player = Some(s),
            Err(e) => self.status = format!("Couldn't play: {e}"),
        }
    }

    fn takes(&self) -> Vec<voice::Take> {
        let mut takes = Vec::new();
        for (i, text) in self.prompts.iter().enumerate() {
            if self.prompt_done[i] {
                takes.push(voice::Take { wav: prompt_wav(i), text: text.clone() });
            }
        }
        for line in &self.lines {
            let wav = format!("{}/mine.wav", line_dir(line));
            if Path::new(&wav).exists() {
                takes.push(voice::Take { wav, text: line.clone() });
            }
        }
        for (id, _, word, _) in WORDS {
            let wav = vowel_file(id, "wav");
            if Path::new(&wav).exists() {
                takes.push(voice::Take { wav, text: word.to_string() });
            }
        }
        takes
    }

    fn start_build(&mut self, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let takes = self.takes();
        let ctx = ctx.clone();
        self.busy = Some(rx);
        self.status = "Building the voice.".to_string();
        std::thread::spawn(move || {
            let status_tx = std::sync::Mutex::new(tx.clone());
            let status_ctx = ctx.clone();
            let status = move |s: &str| {
                let _ = status_tx.lock().unwrap().send(Event::Status(s.to_string()));
                status_ctx.request_repaint();
            };
            let work = format!("{}/voice-work", lab_dir());
            let result = voice::build(&takes, &work, &voice_path(), &status);
            let _ = std::fs::remove_dir_all(&work);
            let _ = tx.send(Event::Built(result));
            ctx.request_repaint();
        });
    }

    fn step_prompt(&self, from: usize, forward: bool) -> Option<usize> {
        let none = vec![false; self.prompts.len()];
        next_prompt(if self.skip_old { &self.prompt_old } else { &none }, from, forward)
    }

    fn toggle_prompt_recording(&mut self) {
        if self.recording_prompt.is_some() {
            let (Some(r), Some(i)) = (self.recording.take(), self.recording_prompt.take()) else { return };
            let x = r.stop();
            let peak = x.iter().fold(0f32, |p, v| p.max(v.abs()));
            match pipeline::write_wav(&prompt_wav(i), &x) {
                Ok(()) => {
                    self.prompt_done[i] = true;
                    self.status = if peak >= 0.99 {
                        "That one clipped. Move back from the mic and record it again.".to_string()
                    } else if peak < 0.05 {
                        "That one was very quiet. Speak up a bit, and record it again.".to_string()
                    } else {
                        String::new()
                    };
                    // a good take moves on by itself; a bad one stays so it can be redone
                    if self.status.is_empty() {
                        if let Some(next) = self.step_prompt(i, true) {
                            self.prompt = next;
                        }
                    }
                }
                Err(e) => self.status = format!("Couldn't save: {e}"),
            }
        } else if self.recording.is_none() && self.busy.is_none() && !self.prompts.is_empty() {
            self.start_recording();
            if self.recording.is_some() {
                self.recording_prompt = Some(self.prompt);
                self.status = "Recording. Read the sentence, then press Space.".to_string();
            }
        }
    }

    fn read_page(&mut self, ctx: &egui::Context) {
        let locked = self.recording.is_some() || self.busy.is_some();
        // not key_pressed: it counts key repeats, so a slightly long press would start and stop a take
        if ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { key: egui::Key::Space, pressed: true, repeat: false, .. }))) {
            self.toggle_prompt_recording();
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Read");
            if self.prompts.is_empty() {
                ui.label("No sentences found: out/voice-lab/prompts.txt is missing.");
                return;
            }
            let done = self.prompt_done.iter().filter(|d| **d).count();
            ui.label(format!("{done} of {} sentences recorded. Press Space to start, read the sentence normally, press Space to stop. It moves on by itself.", self.prompts.len()));
            ui.add_space(14.0);
            ui.label(egui::RichText::new(&self.prompts[self.prompt]).size(30.0));
            ui.add_space(14.0);
            ui.horizontal(|ui| {
                let label = if self.recording_prompt.is_some() { "■ Stop (Space)" } else { "● Record (Space)" };
                if ui.add_enabled(self.busy.is_none(), egui::Button::new(egui::RichText::new(label).size(20.0))).clicked() {
                    self.toggle_prompt_recording();
                }
                if let Some(r) = &self.recording {
                    ui.add(egui::ProgressBar::new(r.peak().sqrt()).desired_width(260.0).text(format!("{:.1} s", r.seconds())));
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let (back, next) = (self.step_prompt(self.prompt, false), self.step_prompt(self.prompt, true));
                if ui.add_enabled(!locked && back.is_some(), egui::Button::new("◀ Back")).clicked() {
                    self.prompt = back.unwrap_or(self.prompt);
                    self.status.clear();
                }
                if ui.add_enabled(!locked && next.is_some(), egui::Button::new("Next ▶")).clicked() {
                    self.prompt = next.unwrap_or(self.prompt);
                    self.status.clear();
                }
                if ui.add_enabled(!locked && self.prompt_done[self.prompt], egui::Button::new("▶ Play this take")).clicked() {
                    self.play_path(&prompt_wav(self.prompt));
                }
                ui.label(format!("Sentence {}", self.prompt + 1));
                ui.checkbox(&mut self.skip_old, "Skip sentences from earlier sessions");
            });
            ui.add_space(14.0);
            ui.separator();
            ui.horizontal(|ui| {
                let takes = self.takes().len();
                if ui.add_enabled(!locked && takes > 0, egui::Button::new(egui::RichText::new("Build voice").size(18.0))).clicked() {
                    self.start_build(ui.ctx());
                }
                ui.label(format!("Uses all {takes} recordings so far, including your Lines and Words."));
                if self.busy.is_some() {
                    ui.spinner();
                }
            });
            ui.add_space(6.0);
            ui.label(&self.status);
        });
    }

    fn save_characters(&self) {
        let _ = serde_json::to_string_pretty(&self.characters).map(|t| std::fs::write(characters_path(), t));
    }

    fn say(&mut self) {
        if self.g2p.is_none() {
            match pipeline::load_g2p() {
                Ok(g) => self.g2p = Some(g),
                Err(e) => {
                    self.speak_info = e;
                    return;
                }
            }
        }
        let (Some(g2p), Some(loaded)) = (&self.g2p, &self.loaded) else { return };
        let c = &self.characters[self.character];
        match voice::speak(g2p, loaded, &self.say_text, c) {
            Ok(spoken) => {
                let x = voice::to_float_at_fs(&spoken.samples);
                let _ = pipeline::write_wav(&format!("{}/last-spoken.wav", lab_dir()), &x);
                self.speak_info = format!(
                    "{:.1} s of speech | {} of {} joins substituted | ROM {} KB ({} joins) | RAM about {} bytes | CPU about {:.1}% of an N64 (estimate)",
                    spoken.samples.len() as f64 / n64::OUT_RATE as f64,
                    spoken.stats.borrowed_joins,
                    spoken.stats.joins,
                    loaded.rom_bytes / 1024,
                    loaded.voice.joins(),
                    n64::ram_bytes(spoken.events),
                    voice::cpu_estimate_percent(&spoken.stats)
                );
                // a short gap, then "Keep reading" moves on
                self.speak_until = Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(x.len() as f64 / pipeline::FS + 0.2));
                self.player = None;
                match audio::play(&x) {
                    Ok(s) => self.player = Some(s),
                    Err(e) => self.speak_info = format!("Couldn't play: {e}"),
                }
                self.spoken_words = spoken.words;
                self.selection = None;
                self.spoken_text = self.say_text.clone();
                self.play_offset = 0.0;
                self.play_end = x.len() as f64 / pipeline::FS;
                self.play_started = Some(std::time::Instant::now());
                self.spoken_audio = x;
            }
            Err(e) => {
                self.speak_info = e;
                // a sentence it can't say shouldn't end "Keep reading"
                self.speak_until = Some(std::time::Instant::now());
            }
        }
    }

    /// Where playback is, in seconds into the last spoken audio, while it's playing.
    fn play_position(&self) -> Option<f64> {
        let p = self.play_offset + self.play_started?.elapsed().as_secs_f64();
        (p < self.play_end).then_some(p)
    }

    /// Exactly the sound the sentence used for this word, with a short fade so the cut doesn't click.
    fn play_words(&mut self, first: usize, last: usize) {
        let (start, end) = (self.spoken_words[first].start, self.spoken_words[last].end + 0.01);
        let to_sample = |t: f64| ((t * pipeline::FS) as usize).min(self.spoken_audio.len());
        let mut x = self.spoken_audio[to_sample(start)..to_sample(end)].to_vec();
        let fade = (0.005 * pipeline::FS) as usize;
        let n = x.len();
        for k in 0..fade.min(n / 2) {
            let g = k as f32 / fade as f32;
            x[k] *= g;
            x[n - 1 - k] *= g;
        }
        self.read_on = false;
        self.speak_until = None;
        self.player = None;
        match audio::play(&x) {
            Ok(s) => {
                self.player = Some(s);
                self.play_offset = start;
                self.play_end = end;
                self.play_started = Some(std::time::Instant::now());
            }
            Err(e) => self.speak_info = format!("Couldn't play: {e}"),
        }
    }

    fn save_note(&mut self, a: usize, b: usize) {
        let words: Vec<&str> = self.spoken_words[a..=b].iter().map(|w| w.text.as_str()).collect();
        let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        self.notes.push(voice::Note {
            sentence: self.spoken_text.clone(),
            words: words.join(" "),
            note: self.note_text.trim().to_string(),
            character: self.characters[self.character].name.clone(),
            start: self.spoken_words[a].start,
            end: self.spoken_words[b].end,
            when,
            fixed: false,
        });
        self.note_text.clear();
        self.save_notes();
    }

    fn save_notes(&self) {
        let lines: Vec<String> = self.notes.iter().filter_map(|n| serde_json::to_string(n).ok()).collect();
        let _ = std::fs::write(notes_path(), lines.join("\n") + "\n");
    }

    fn book_step(&mut self, delta: isize) {
        self.book_pos = self.book_pos.saturating_add_signed(delta).min(self.book.len() - 1);
        self.say_text = self.book[self.book_pos].to_string();
    }

    fn speak_page(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Speak");
            if self.loaded.is_none() {
                ui.label("No voice yet. Record some sentences on the Read tab, then press Build voice.");
                return;
            }
            ui.add(egui::TextEdit::multiline(&mut self.say_text).font(egui::TextStyle::Heading).desired_rows(3).desired_width(f32::INFINITY));
            ui.add_space(6.0);
            let mut speak_now = false;
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(egui::RichText::new("▶ Speak").size(20.0))).clicked() {
                    speak_now = true;
                }
                if ui.add_enabled(self.player.is_some(), egui::Button::new("Stop")).clicked() {
                    self.player = None;
                    self.read_on = false;
                    self.speak_until = None;
                    self.play_started = None;
                }
                ui.separator();
                ui.label("Character");
                let shown = self.characters[self.character].name.clone();
                egui::ComboBox::from_id_salt("character").selected_text(shown).show_ui(ui, |ui| {
                    for (i, c) in self.characters.iter().enumerate() {
                        ui.selectable_value(&mut self.character, i, c.name.clone());
                    }
                });
                if ui.button("Copy as new").clicked() {
                    let mut c = self.characters[self.character].clone();
                    c.name = format!("Character {}", self.characters.len() + 1);
                    self.characters.push(c);
                    self.character = self.characters.len() - 1;
                    self.save_characters();
                }
                if ui.add_enabled(self.characters.len() > 1, egui::Button::new("Delete")).clicked() {
                    self.characters.remove(self.character);
                    self.character = self.character.min(self.characters.len() - 1);
                    self.save_characters();
                }
                if ui.button("Reset all").clicked() {
                    self.characters = voice::Character::presets();
                    self.character = 0;
                    self.save_characters();
                }
            });
            if !self.spoken_words.is_empty() {
                ui.add_space(8.0);
                let now = self.play_position();
                if now.is_some() {
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(30));
                }
                let mut clicked = None;
                let selection = self.selection;
                let noted = notes_on_words(&self.notes, &self.spoken_text, &self.spoken_words);
                ui.horizontal_wrapped(|ui| {
                    for (i, w) in self.spoken_words.iter().enumerate() {
                        let mut text = egui::RichText::new(&w.text).size(22.0);
                        if now.is_some_and(|t| t >= w.start && t < w.end) {
                            text = text.background_color(egui::Color32::from_rgb(255, 214, 90)).color(egui::Color32::BLACK);
                        } else if selection.is_some_and(|(a, b)| i >= a && i <= b) {
                            text = text.background_color(egui::Color32::from_rgb(170, 205, 255)).color(egui::Color32::BLACK);
                        } else if !noted[i].is_empty() {
                            text = text.background_color(egui::Color32::from_rgb(255, 190, 190)).color(egui::Color32::BLACK);
                        }
                        let mut label = ui.add(egui::Label::new(text).sense(egui::Sense::click()).selectable(false)).on_hover_cursor(egui::CursorIcon::PointingHand);
                        if !noted[i].is_empty() {
                            let notes: Vec<&str> = noted[i].iter().map(|&n| self.notes[n].note.as_str()).collect();
                            label = label.on_hover_text(notes.join("\n"));
                        }
                        if label.clicked() {
                            clicked = Some(i);
                        }
                    }
                });
                ui.weak("Click a word to hear it. Shift-click another to take the whole stretch. Pink words have an open note: click one to edit it, then shift-click it to add another.");
                if let Some(i) = clicked {
                    let shift = ui.input(|k| k.modifiers.shift);
                    let (a, b) = match self.selection {
                        Some((a, _)) if shift => (a.min(i), a.max(i)),
                        _ => (i, i),
                    };
                    self.selection = Some((a, b));
                    // a plain click opens the notes already on the word; shift-click starts a new one
                    self.note_edits = if shift { Vec::new() } else { noted[i].iter().map(|&n| (n, self.notes[n].note.clone())).collect() };
                    self.play_words(a, b);
                }
                let mut saved_edit = None;
                for (row, (n, text)) in self.note_edits.iter_mut().enumerate() {
                    let note = &self.notes[*n];
                    let can_save = !text.trim().is_empty() && text.trim() != note.note;
                    ui.horizontal(|ui| {
                        ui.label(format!("Note on \"{}\":", note.words));
                        let field = ui.add(egui::TextEdit::singleline(text).hint_text("what sounds wrong?").desired_width(360.0));
                        let enter = field.lost_focus() && ui.input(|k| k.key_pressed(egui::Key::Enter));
                        if ui.add_enabled(can_save, egui::Button::new("Save note")).clicked() || (enter && can_save) {
                            saved_edit = Some(row);
                        }
                    });
                }
                if let Some(row) = saved_edit {
                    let (n, text) = &self.note_edits[row];
                    self.notes[*n].note = text.trim().to_string();
                    self.save_notes();
                }
                if let (Some((a, b)), true) = (self.selection, self.note_edits.is_empty()) {
                    let mut save = false;
                    ui.horizontal(|ui| {
                        let words: Vec<&str> = self.spoken_words[a..=b].iter().map(|w| w.text.as_str()).collect();
                        ui.label(format!("Note on \"{}\":", words.join(" ")));
                        let field = ui.add(egui::TextEdit::singleline(&mut self.note_text).hint_text("what sounds wrong?").desired_width(360.0));
                        save |= field.lost_focus() && ui.input(|k| k.key_pressed(egui::Key::Enter));
                        save |= ui.add_enabled(!self.note_text.trim().is_empty(), egui::Button::new("Save note")).clicked();
                    });
                    if save && !self.note_text.trim().is_empty() {
                        self.save_note(a, b);
                    }
                }
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("Alice in Wonderland");
                if ui.button("◀ Previous").clicked() {
                    self.book_step(-1);
                    speak_now = true;
                }
                if ui.button("Next ▶").clicked() {
                    self.book_step(1);
                    speak_now = true;
                }
                let last = self.book.len() - 1;
                if ui.add(egui::Slider::new(&mut self.book_pos, 0..=last).text("sentence")).changed() {
                    self.book_step(0);
                }
                ui.checkbox(&mut self.read_on, "Keep reading");
            });
            ui.add_space(10.0);
            let mut changed = false;
            let c = &mut self.characters[self.character];
            egui::Grid::new("sliders").spacing([14.0, 10.0]).show(ui, |ui| {
                ui.label("Name");
                changed |= ui.text_edit_singleline(&mut c.name).changed();
                ui.end_row();
                let mut row = |ui: &mut egui::Ui, label: &str, value: &mut f64, range: std::ops::RangeInclusive<f64>, hint: &str| {
                    ui.label(label);
                    changed |= ui.add(egui::Slider::new(value, range).fixed_decimals(2)).changed();
                    ui.weak(hint);
                    ui.end_row();
                };
                row(ui, "Pitch (Hz)", &mut c.pitch, 50.0..=400.0, "how high the voice sits");
                row(ui, "Expression", &mut c.range, 0.0..=2.5, "flat to sing-song");
                row(ui, "Speed", &mut c.rate, 0.5..=2.0, "slow to fast");
                row(ui, "Size", &mut c.size, 0.7..=1.4, "big body to small body");
                row(ui, "Breathiness", &mut c.breath, 0.0..=1.0, "clear to whispery");
                row(ui, "Brightness", &mut c.bright, -0.6..=0.8, "muffled to sharp");
                row(ui, "Roughness", &mut c.rough, 0.0..=1.0, "smooth to gravelly");
            });
            if changed {
                self.save_characters();
            }
            if !self.notes.is_empty() {
                ui.add_space(10.0);
                let (mut go_to, mut remove, mut toggle) = (None, None, None);
                let open = self.notes.iter().filter(|n| !n.fixed).count();
                let fixed = self.notes.len() - open;
                egui::CollapsingHeader::new(format!("Notes ({open} open, {fixed} fixed)")).id_salt("notes").default_open(true).show(ui, |ui| {
                    ui.checkbox(&mut self.show_fixed, "Show fixed");
                    // all the room the window has, less the two status lines under the list
                    egui::ScrollArea::vertical().max_height((ui.available_height() - 70.0).max(140.0)).show(ui, |ui| {
                        for (i, n) in self.notes.iter().enumerate() {
                            if n.fixed && !self.show_fixed {
                                continue;
                            }
                            ui.horizontal_wrapped(|ui| {
                                if n.fixed {
                                    ui.label(egui::RichText::new(&n.words).strikethrough().weak());
                                } else {
                                    ui.strong(&n.words);
                                }
                                ui.label(&n.note);
                                ui.weak(&n.sentence);
                                if ui.small_button("Go to").clicked() {
                                    go_to = Some(i);
                                }
                                if ui.small_button(if n.fixed { "Reopen" } else { "Fixed" }).clicked() {
                                    toggle = Some(i);
                                }
                                if ui.small_button("Delete").clicked() {
                                    remove = Some(i);
                                }
                            });
                        }
                    });
                });
                if let Some(i) = toggle {
                    self.notes[i].fixed = !self.notes[i].fixed;
                    self.save_notes();
                }
                if let Some(i) = go_to {
                    self.say_text = self.notes[i].sentence.clone();
                    speak_now = true;
                }
                if let Some(i) = remove {
                    self.notes.remove(i);
                    // the indices of any notes open for editing have just moved
                    self.note_edits.clear();
                    self.save_notes();
                }
            }
            ui.add_space(10.0);
            ui.separator();
            ui.label(&self.speak_info);
            ui.weak("The C library is the code that would go on the N64; the text-to-sounds step still runs on your PC.");
            if speak_now {
                self.say();
            }
        });
    }

    fn start_fit(&mut self, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let body = if self.body == 2 { "female" } else { "male" };
        let ctx = ctx.clone();
        self.busy = Some(rx);
        self.status = "Fitting your vowels.".to_string();
        std::thread::spawn(move || {
            let _ = tx.send(Event::Fitted(fit::fit_recordings(body)));
            ctx.request_repaint();
        });
    }

    fn start_vowel(&mut self, i: usize) {
        self.start_recording();
        if self.recording.is_some() {
            self.recording_vowel = Some(i);
            self.status = format!("Say \"{}\", then press Stop.", WORDS[i].2);
        }
    }

    fn stop_vowel(&mut self) {
        let (Some(r), Some(i)) = (self.recording.take(), self.recording_vowel.take()) else { return };
        let x = r.stop();
        let (id, _, word, onset) = WORDS[i];
        if let Err(e) = pipeline::write_wav(&vowel_file(id, "wav"), &x) {
            self.status = format!("Couldn't save: {e}");
            return;
        }
        let peak = x.iter().fold(0f32, |p, v| p.max(v.abs()));
        match pipeline::measure_word(&x, onset) {
            Some(h) => {
                let _ = serde_json::to_string_pretty(&h).map(|t| std::fs::write(vowel_file(id, "json"), t));
                self.held[i] = Some(h);
                self.status = if peak >= 0.99 {
                    format!("Saved \"{word}\", but it clipped. Move back from the mic and say it again.")
                } else if peak < 0.05 {
                    format!("Saved \"{word}\", but it's very quiet. Say it again a bit louder or closer to the mic.")
                } else {
                    format!("Saved \"{word}\".")
                };
            }
            None => {
                let _ = std::fs::remove_file(vowel_file(id, "json"));
                self.held[i] = None;
                self.status = format!("Couldn't find the vowel in \"{word}\". Try saying it again, a little louder.");
            }
        }
    }

    fn vowels_page(&mut self, ctx: &egui::Context) {
        let locked = self.recording.is_some() || self.busy.is_some();
        let body = if self.body == 2 { "female" } else { "male" };
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Words");
            ui.label("Press Record, say the word once the way you normally would, then press Stop. Record it again if you stumble.");
            ui.add_space(4.0);
            ui.add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Microphone");
                    let shown = self.input.clone().unwrap_or_else(|| "none".to_string());
                    egui::ComboBox::from_id_salt("mic-vowels").selected_text(shown).width(320.0).show_ui(ui, |ui| {
                        for name in &self.inputs {
                            ui.selectable_value(&mut self.input, Some(name.clone()), name);
                        }
                    });
                });
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if let Some(r) = &self.recording {
                    ui.add(egui::ProgressBar::new(r.peak().sqrt()).desired_width(260.0).text(format!("{:.1} s", r.seconds())));
                }
                ui.label(&self.status);
            });
            ui.separator();
            let (mut record, mut stop, mut play) = (None, false, None);
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("held").striped(true).spacing([16.0, 6.0]).show(ui, |ui| {
                    for head in ["say", "", "", "you F1", "F2", "F3", "engine F1", "F2", "F3", "fitted F1", "F2", "F3"] {
                        ui.strong(head);
                    }
                    ui.end_row();
                    for (i, (id, ipa, word, _)) in WORDS.iter().enumerate() {
                        ui.label(egui::RichText::new(*word).size(20.0));
                        if self.recording_vowel == Some(i) {
                            if ui.button("■ Stop").clicked() {
                                stop = true;
                            }
                        } else if ui.add_enabled(!locked, egui::Button::new("● Record")).clicked() {
                            record = Some(i);
                        }
                        if ui.add_enabled(!locked && Path::new(&vowel_file(id, "wav")).exists(), egui::Button::new("▶")).clicked() {
                            play = Some(i);
                        }
                        let h = self.held[i];
                        for k in 0..3 {
                            ui.label(hz(h.map(|v| v.formants[k])));
                        }
                        let e = pipeline::engine_formants(&self.vowel_table, body, ipa);
                        for k in 0..3 {
                            ui.label(hz(e.map(|v| v[k])));
                        }
                        let fitted = self.fit.as_ref().and_then(|f| f.get(ipa));
                        for k in 0..3 {
                            ui.label(hz(fitted.map(|v| v.achieved[k])));
                        }
                        ui.end_row();
                    }
                });
            });
            ui.add_space(4.0);
            let done = self.held.iter().filter(|h| h.is_some()).count();
            ui.horizontal(|ui| {
                ui.label(format!("{done} of {} words done", WORDS.len()));
                if ui.add_enabled(!locked && done > 0, egui::Button::new("Fit my vowels")).clicked() {
                    self.start_fit(ui.ctx());
                }
                if self.busy.is_some() {
                    ui.spinner();
                }
            });
            if let Some(i) = record {
                self.start_vowel(i);
            }
            if stop {
                self.stop_vowel();
            }
            if let Some(i) = play {
                self.play_path(&vowel_file(WORDS[i].0, "wav"));
            }
        });
    }

    fn compare(&mut self, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let (wav, text, dir) = (self.file("mine.wav"), self.text().to_string(), line_dir(self.text()));
        let body = BODIES[self.body].to_string();
        let accent = ACCENTS[self.accent].0.to_string();
        let ctx = ctx.clone();
        self.busy = Some(rx);
        self.report = None;
        self.status = "Starting".to_string();
        std::thread::spawn(move || {
            let (status_tx, status_ctx) = (tx.clone(), ctx.clone());
            let status = move |s: &str| {
                let _ = status_tx.send(Event::Status(s.to_string()));
                status_ctx.request_repaint();
            };
            let result = pipeline::analyze(&wav, &text, &body, &accent, &dir, &status);
            let _ = tx.send(Event::Done(result));
            ctx.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(rx) = &self.busy else { return };
        let mut done = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                Event::Status(s) => self.status = s,
                Event::Done(Ok(report)) => {
                    self.status = "Done. Compare the three versions by ear.".to_string();
                    self.report = Some(report);
                    done = true;
                }
                Event::Done(Err(e)) | Event::Fitted(Err(e)) => {
                    self.status = format!("Failed: {e}");
                    done = true;
                }
                Event::Built(Ok(r)) => {
                    self.status = format!("Voice built from {} recordings: {} sound joins, {} KB. Open the Speak tab to try it.", r.takes, r.joins, r.rom_bytes / 1024);
                    self.loaded = voice::load(&voice_path()).ok();
                    done = true;
                }
                Event::Built(Err(e)) => {
                    self.status = format!("Couldn't build the voice: {e}");
                    done = true;
                }
                Event::Fitted(Ok(f)) => {
                    let e = f.error_pct();
                    self.status = format!("Fitted {} vowels (F1 within {:.1}%, F2 {:.1}%, F3 {:.1}% on average). On the Lines tab, press Compare on a line to hear it.", f.vowels.len(), e[0], e[1], e[2]);
                    self.fit = Some(f);
                    done = true;
                }
            }
        }
        if done {
            self.busy = None;
        }
    }
}

fn hz(v: Option<f64>) -> String {
    v.map_or("-".to_string(), |x| format!("{x:.0}"))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if self.recording.is_some() {
            self.read_on = false;
        }
        if self.read_on {
            if self.speak_until.is_some_and(|t| std::time::Instant::now() >= t) {
                if self.book_pos + 1 < self.book.len() {
                    self.book_step(1);
                    self.say();
                } else {
                    self.read_on = false;
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let locked = self.recording.is_some() || self.busy.is_some();
        if self.recording.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    let lines = ui.selectable_value(&mut self.page, Page::Lines, "Lines").clicked();
                    let words = ui.selectable_value(&mut self.page, Page::Vowels, "Words").clicked();
                    let read = ui.selectable_value(&mut self.page, Page::Read, "Read").clicked();
                    let speak = ui.selectable_value(&mut self.page, Page::Speak, "Speak").clicked();
                    if lines || words || read || speak {
                        self.status.clear();
                    }
                });
            });
        });
        // a take in progress belongs to its sentence: another tab's Stop would save it over a Line
        if self.recording_prompt.is_some() {
            self.page = Page::Read;
        }
        if self.page == Page::Vowels {
            self.vowels_page(ctx);
            return;
        }
        if self.page == Page::Read {
            if self.recording.is_some() {
                ctx.request_repaint_after(std::time::Duration::from_millis(33));
            }
            self.read_page(ctx);
            return;
        }
        if self.page == Page::Speak {
            self.speak_page(ctx);
            return;
        }

        egui::SidePanel::left("lines").default_width(300.0).show(ctx, |ui| {
            ui.heading("Lines");
            ui.add_space(4.0);
            let mut pick = None;
            egui::ScrollArea::vertical().max_height(ui.available_height() - 40.0).show(ui, |ui| {
                for (i, line) in self.lines.iter().enumerate() {
                    let recorded = Path::new(&format!("{}/mine.wav", line_dir(line))).exists();
                    let label = format!("{} {line}", if recorded { "●" } else { "○" });
                    if ui.add_enabled(!locked, egui::Button::new(label).selected(i == self.selected).wrap()).clicked() {
                        pick = Some(i);
                    }
                }
            });
            if let Some(i) = pick {
                self.selected = i;
                self.status = String::new();
                self.load_report();
            }
            ui.separator();
            ui.add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Add line").clicked() {
                        self.lines.push("Type a new line here.".to_string());
                        self.selected = self.lines.len() - 1;
                        self.report = None;
                        self.save_lines();
                    }
                    if ui.add_enabled(self.lines.len() > 1, egui::Button::new("Remove from list")).clicked() {
                        self.lines.remove(self.selected);
                        self.selected = self.selected.min(self.lines.len() - 1);
                        self.save_lines();
                        self.load_report();
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                let edit = ui.add(egui::TextEdit::multiline(&mut self.lines[self.selected]).font(egui::TextStyle::Heading).desired_rows(2).desired_width(f32::INFINITY));
                if edit.changed() {
                    self.report = None;
                    self.save_lines();
                }
                ui.horizontal(|ui| {
                    ui.label("Microphone");
                    let shown = self.input.clone().unwrap_or_else(|| "none".to_string());
                    egui::ComboBox::from_id_salt("mic").selected_text(shown).width(320.0).show_ui(ui, |ui| {
                        for name in &self.inputs {
                            ui.selectable_value(&mut self.input, Some(name.clone()), name);
                        }
                    });
                    ui.label("Voice body");
                    egui::ComboBox::from_id_salt("body").selected_text(BODIES[self.body]).show_ui(ui, |ui| {
                        for (i, b) in BODIES.iter().enumerate() {
                            ui.selectable_value(&mut self.body, i, *b);
                        }
                    });
                    ui.label("Accent");
                    egui::ComboBox::from_id_salt("accent").selected_text(ACCENTS[self.accent].1).show_ui(ui, |ui| {
                        for (i, (_, name)) in ACCENTS.iter().enumerate() {
                            ui.selectable_value(&mut self.accent, i, *name);
                        }
                    });
                });
            });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                if let Some(r) = &self.recording {
                    let (secs, peak) = (r.seconds(), r.peak());
                    if ui.add(egui::Button::new(egui::RichText::new("■ Stop").size(20.0).color(egui::Color32::LIGHT_RED))).clicked() {
                        self.stop_recording();
                    } else {
                        ui.add(egui::ProgressBar::new(peak.sqrt()).desired_width(260.0).text(format!("{secs:.1} s")));
                    }
                } else if ui.add_enabled(self.busy.is_none(), egui::Button::new(egui::RichText::new("● Record").size(20.0))).clicked() {
                    self.start_recording();
                }
            });
            ui.add_space(6.0);

            let idle = !locked;
            ui.horizontal(|ui| {
                let (mine, engine, timed) = (self.has("mine.wav"), self.has("engine.wav") && self.report.is_some(), self.has("engine-my-timing.wav") && self.report.is_some());
                if ui.add_enabled(idle && mine, egui::Button::new("▶ You")).clicked() {
                    self.play("mine.wav");
                }
                if ui.add_enabled(idle && engine, egui::Button::new("▶ Engine")).clicked() {
                    self.play("engine.wav");
                }
                if ui.add_enabled(idle && timed, egui::Button::new("▶ Engine, your timing and pitch")).clicked() {
                    self.play("engine-my-timing.wav");
                }
                if ui.add_enabled(idle && self.has("engine-my-vowels.wav") && self.report.is_some(), egui::Button::new("▶ Engine, your vowels")).clicked() {
                    self.play("engine-my-vowels.wav");
                }
                if ui.add_enabled(idle && self.player.is_some(), egui::Button::new("Stop playback")).clicked() {
                    self.player = None;
                }
                ui.separator();
                if ui.add_enabled(idle && mine, egui::Button::new("Compare")).clicked() {
                    self.compare(ctx);
                }
            });
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                if self.busy.is_some() {
                    ui.spinner();
                }
                ui.label(&self.status);
            });
            ui.separator();

            if let Some(r) = &self.report {
                ui.label(egui::RichText::new(r.summary()).monospace());
                ui.add_space(6.0);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("vowels").striped(true).spacing([18.0, 4.0]).show(ui, |ui| {
                        for head in ["word", "vowel", "you ms", "engine ms", "you F1", "engine F1", "you F2", "engine F2", "you F3", "engine F3", "fitted F1", "F2", "F3"] {
                            ui.strong(head);
                        }
                        ui.end_row();
                        for v in &r.vowels {
                            ui.label(&v.word);
                            ui.label(format!("{}{}", if v.stress == 1 { "ˈ" } else { "" }, v.ipa));
                            ui.label(format!("{:.0}", v.mine_ms));
                            ui.label(format!("{:.0}", v.engine_ms));
                            for k in 0..3 {
                                ui.label(hz(v.mine.map(|f| f[k])));
                                ui.label(hz(v.engine.map(|f| f[k])));
                            }
                            for k in 0..3 {
                                ui.label(hz(v.fitted.map(|f| f[k])));
                            }
                            ui.end_row();
                        }
                    });
                });
            }
        });
    }
}

/// egui's bundled fonts may not cover IPA; Windows' Segoe UI does.
fn install_fonts(ctx: &egui::Context) {
    let Ok(bytes) = std::fs::read("C:/Windows/Fonts/segoeui.ttf") else { return };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("segoe".to_string(), Arc::new(egui::FontData::from_owned(bytes)));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("segoe".to_string());
    }
    ctx.set_fonts(fonts);
}

#[cfg(windows)]
unsafe extern "system" {
    fn AttachConsole(process_id: u32) -> i32;
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(windows)]
    if args.first().is_some_and(|a| a != "--words" && a != "--speak" && a != "--read") {
        const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
        // SAFETY: takes no pointers; failing just means there is no terminal to print to
        unsafe { AttachConsole(ATTACH_PARENT_PROCESS) };
    }
    match args.first().map(String::as_str) {
        Some("--check-audio") => {
            audio::check_devices();
            return Ok(());
        }
        Some("--self-test") => return pipeline::self_test(),
        Some("--select-prompts") => {
            let (pool, out, n) = (args.get(1).ok_or("pool file?")?, args.get(2).ok_or("output file?")?, args.get(3).and_then(|n| n.parse().ok()).unwrap_or(150));
            return voice::select_prompts(pool, out, n);
        }
        Some("--build-voice") => {
            // a list of "wav<TAB>text" lines, then the voice file to write
            let (list, out) = (args.get(1).ok_or("list file?")?, args.get(2).ok_or("output file?")?);
            let text = std::fs::read_to_string(list).map_err(|e| format!("{list}: {e}"))?;
            let takes: Vec<voice::Take> = text.lines().filter_map(|l| l.split_once('\t')).map(|(w, t)| voice::Take { wav: w.to_string(), text: t.to_string() }).collect();
            let work = format!("{out}.work");
            let r = voice::build(&takes, &work, out, &|s| println!("... {s}"))?;
            let _ = std::fs::remove_dir_all(&work);
            println!("{r:?}");
            return Ok(());
        }
        Some("--ladder") => {
            // a list of "wav<TAB>text" lines, the voice, and where to write the four stages
            let (list, file, out) = (args.get(1).ok_or("list file?")?, args.get(2).ok_or("voice file?")?, args.get(3).ok_or("output folder?")?);
            let text = std::fs::read_to_string(list).map_err(|e| format!("{list}: {e}"))?;
            let takes: Vec<voice::Take> = text.lines().filter_map(|l| l.split_once('\t')).map(|(w, t)| voice::Take { wav: w.to_string(), text: t.to_string() }).collect();
            return voice::ladder(&takes, file, out);
        }
        Some("--book-check") => {
            // every sentence of the book through the synth: what breaks, what clips
            let loaded = voice::load(args.get(1).ok_or("voice file?")?)?;
            let g2p = pipeline::load_g2p()?;
            let c = voice::Character::presets().remove(0);
            let (mut samples, mut clipped, mut joins, mut borrowed, mut failed) = (0u64, 0u64, 0u64, 0u64, 0usize);
            let started = std::time::Instant::now();
            let lines: Vec<&str> = BOOK.lines().collect();
            for line in &lines {
                match voice::speak(&g2p, &loaded, line, &c) {
                    Ok(s) => {
                        samples += s.samples.len() as u64;
                        clipped += s.samples.iter().filter(|v| v.unsigned_abs() >= 32767).count() as u64;
                        joins += s.stats.joins as u64;
                        borrowed += s.stats.borrowed_joins as u64;
                    }
                    Err(e) => {
                        failed += 1;
                        if failed <= 15 {
                            println!("failed: {e}: {line}");
                        }
                    }
                }
            }
            let seconds = samples as f64 / n64::OUT_RATE as f64;
            println!("{} sentences, {failed} failed; {:.1} minutes of speech in {:.1} s", lines.len(), seconds / 60.0, started.elapsed().as_secs_f64());
            println!("joins {joins}, substituted {borrowed} ({:.2}%); clipped samples {clipped} ({:.4}%)", 100.0 * borrowed as f64 / joins.max(1) as f64, 100.0 * clipped as f64 / samples.max(1) as f64);
            return Ok(());
        }
        Some("--select-targeted") => {
            let need = |i: usize, what: &'static str| args.get(i).ok_or(what);
            let n = args.get(5).and_then(|n| n.parse().ok()).unwrap_or(100);
            return voice::select_targeted(need(1, "book?")?, need(2, "recorded list?")?, need(3, "sentences to leave out?")?, need(4, "output file?")?, n);
        }
        Some("--explain") => {
            // what the synth does for a piece of text, sound by sound
            let loaded = voice::load(args.get(1).ok_or("voice file?")?)?;
            let g2p = pipeline::load_g2p()?;
            let c = voice::Character::presets().remove(0);
            print!("{}", voice::explain(&g2p, &loaded, args.get(2).ok_or("text?")?, &c)?);
            return Ok(());
        }
        Some("--say") => {
            let (file, text, out) = (args.get(1).ok_or("voice file?")?, args.get(2).ok_or("text?")?, args.get(3).ok_or("output wav?")?);
            let name = args.get(4).map_or("You", String::as_str);
            let loaded = voice::load(file)?;
            let g2p = pipeline::load_g2p()?;
            let c = voice::Character::presets().into_iter().find(|c| c.name == name).ok_or("no such character")?;
            let spoken = voice::speak(&g2p, &loaded, text, &c)?;
            pipeline::write_wav(out, &voice::to_float_at_fs(&spoken.samples))?;
            println!("{} samples, {} joins, {} borrowed; ROM {} bytes; RAM about {} bytes; CPU about {:.1}% (estimate)", spoken.samples.len(), spoken.stats.joins, spoken.stats.borrowed_joins, loaded.rom_bytes, n64::ram_bytes(spoken.events), voice::cpu_estimate_percent(&spoken.stats));
            return Ok(());
        }
        Some("--fit") => {
            let f = fit::fit_recordings(args.get(1).map_or("male", String::as_str))?;
            for v in &f.vowels {
                println!("{:6} {:3} you {:?} fitted {:?}", v.word, v.ipa, v.target.map(f64::round), v.achieved.map(f64::round));
            }
            println!("mean error (%): {:?}", f.error_pct().map(|e| (e * 10.0).round() / 10.0));
            return Ok(());
        }
        Some("--measure") => {
            for path in &args[1..] {
                let x = pipeline::read_wav(path)?;
                let peak = x.iter().fold(0f32, |p, v| p.max(v.abs()));
                println!("{path}: {:.2} s, peak {peak:.3}, {:?}", x.len() as f64 / FS, pipeline::measure_word(&x, false));
            }
            return Ok(());
        }
        Some("--analyze") => {
            let (wav, text, out) = match (args.get(1), args.get(2), args.get(3)) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => return Err("usage: voice-lab --analyze in.wav \"text\" outdir [auto|male|female]".to_string()),
            };
            let body = args.get(4).map_or("auto", String::as_str);
            let accent = args.get(5).map_or("british", String::as_str);
            let report = pipeline::analyze(wav, text, body, accent, out, &|s| println!("... {s}"))?;
            println!("{}", report.summary());
            for v in &report.vowels {
                println!("{:12} {:3} you {:4.0} ms {:?}  engine {:4.0} ms {:?}", v.word, v.ipa, v.mine_ms, v.mine.map(|f| f.map(|x| x.round())), v.engine_ms, v.engine.map(|f| f.map(|x| x.round())));
            }
            return Ok(());
        }
        _ => {}
    }
    chimera::init();
    let open_words = args.first().is_some_and(|a| a == "--words");
    let open_speak = args.first().is_some_and(|a| a == "--speak");
    let open_read = args.first().is_some_and(|a| a == "--read");
    let options =eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_title("CHIMERA Voice Lab").with_inner_size([1120.0, 720.0]), ..Default::default() };
    eframe::run_native(
        "CHIMERA Voice Lab",
        options,
        Box::new(move |cc| {
            install_fonts(&cc.egui_ctx);
            let mut app = App::new();
            if open_words {
                app.page = Page::Vowels;
                app.status.clear();
            }
            if open_speak {
                app.page = Page::Speak;
            }
            if open_read {
                app.page = Page::Read;
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{next_prompt, notes_on_words, voice};

    fn note(sentence: &str, words: &str, start: f64, fixed: bool) -> voice::Note {
        voice::Note { sentence: sentence.into(), words: words.into(), note: "n".into(), character: "You".into(), start, end: start, when: 0, fixed }
    }

    #[test]
    fn clicking_a_word_finds_every_open_note_on_it_to_edit() {
        let words: Vec<voice::SpokenWord> = ["a", "b", "a"].iter().enumerate().map(|(i, t)| voice::SpokenWord { text: t.to_string(), start: i as f64, end: i as f64 + 1.0 }).collect();
        let notes = [
            note("a b a", "a", 2.1, false),  // the later "a", by time
            note("a b a", "a", 1.9, false),  // a second note on that word: both must open
            note("a b a", "b", 1.0, true),   // fixed: a click adds a new note instead
            note("other", "b", 1.0, false),  // another sentence
            note("a b a", "b a", 1.0, false), // a stretch counts for each of its words
        ];
        assert_eq!(notes_on_words(&notes, "a b a", &words), vec![vec![], vec![4], vec![0, 1, 4]]);
    }

    #[test]
    fn reading_on_passes_over_sentences_from_earlier_sessions() {
        // after the one sentence an earlier session missed, the next is the first new one, not its old neighbour
        let skip = [true, false, true, true, false];
        assert_eq!(next_prompt(&skip, 1, true), Some(4));
        assert_eq!(next_prompt(&skip, 4, false), Some(1));
        assert_eq!(next_prompt(&skip, 4, true), None);
        assert_eq!(next_prompt(&skip, 1, false), None);
    }
}

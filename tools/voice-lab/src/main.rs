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
mod pipeline;

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
}

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
        };
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
        let locked = self.recording.is_some() || self.busy.is_some();
        if self.recording.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    let lines = ui.selectable_value(&mut self.page, Page::Lines, "Lines").clicked();
                    let words = ui.selectable_value(&mut self.page, Page::Vowels, "Words").clicked();
                    if lines || words {
                        self.status.clear();
                    }
                });
            });
        });
        if self.page == Page::Vowels {
            self.vowels_page(ctx);
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
    if args.first().is_some_and(|a| a != "--words") {
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
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| e.to_string())
}

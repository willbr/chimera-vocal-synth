//! The microphone and the speakers, through cpal.

use crate::pipeline::{FS, resample};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

pub fn input_names() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices().map(|d| d.filter_map(|x| x.name().ok()).collect()).unwrap_or_default()
}

pub fn default_input_name() -> Option<String> {
    cpal::default_host().default_input_device().and_then(|d| d.name().ok())
}

fn find_input(name: Option<&str>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    match name {
        Some(n) => host.input_devices().map_err(|e| e.to_string())?.find(|d| d.name().is_ok_and(|x| x == n)).ok_or_else(|| format!("microphone {n:?} is gone")),
        None => host.default_input_device().ok_or_else(|| "no microphone found".to_string()),
    }
}

pub struct Recording {
    stream: Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    peak: Arc<AtomicU32>,
    rate: f64,
}

fn input_stream<T>(device: &cpal::Device, config: &cpal::StreamConfig, samples: Arc<Mutex<Vec<f32>>>, peak: Arc<AtomicU32>) -> Result<Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = config.channels.max(1) as usize;
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let mut out = samples.lock().unwrap();
                let mut p = 0f32;
                for frame in data.chunks(ch) {
                    let v = frame.iter().map(|s| f32::from_sample(*s)).sum::<f32>() / ch as f32;
                    p = p.max(v.abs());
                    out.push(v);
                }
                peak.store(p.to_bits(), Ordering::Relaxed);
            },
            |e| eprintln!("microphone error: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

fn open_input(device: &cpal::Device, samples: Arc<Mutex<Vec<f32>>>, peak: Arc<AtomicU32>) -> Result<(Stream, f64), String> {
    let supported = device.default_input_config().map_err(|e| e.to_string())?;
    let format = supported.sample_format();
    let rate = supported.sample_rate().0 as f64;
    let config: cpal::StreamConfig = supported.into();
    let stream = match format {
        SampleFormat::F32 => input_stream::<f32>(device, &config, samples, peak),
        SampleFormat::I16 => input_stream::<i16>(device, &config, samples, peak),
        SampleFormat::I32 => input_stream::<i32>(device, &config, samples, peak),
        SampleFormat::U16 => input_stream::<u16>(device, &config, samples, peak),
        other => Err(format!("unsupported microphone format {other:?}")),
    }?;
    Ok((stream, rate))
}

impl Recording {
    pub fn start(device: Option<&str>) -> Result<Recording, String> {
        let device = find_input(device)?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let peak = Arc::new(AtomicU32::new(0));
        let (stream, rate) = open_input(&device, samples.clone(), peak.clone())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Recording { stream, samples, peak, rate })
    }

    pub fn peak(&self) -> f32 {
        f32::from_bits(self.peak.load(Ordering::Relaxed))
    }

    pub fn seconds(&self) -> f64 {
        self.samples.lock().unwrap().len() as f64 / self.rate
    }

    /// Stop and return the take as mono at the engine's rate.
    pub fn stop(self) -> Vec<f32> {
        drop(self.stream);
        let x = self.samples.lock().unwrap();
        resample(&x, self.rate, FS)
    }
}

fn output_stream<T>(device: &cpal::Device, config: &cpal::StreamConfig, data: Vec<f32>, position: Arc<Mutex<usize>>) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let ch = config.channels.max(1) as usize;
    device
        .build_output_stream(
            config,
            move |out: &mut [T], _| {
                let mut pos = position.lock().unwrap();
                for frame in out.chunks_mut(ch) {
                    let v = data.get(*pos).copied().unwrap_or(0.0);
                    *pos += 1;
                    for s in frame.iter_mut() {
                        *s = T::from_sample(v);
                    }
                }
            },
            |e| eprintln!("speaker error: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

fn open_output(x: &[f32]) -> Result<Stream, String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no speakers found")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let format = supported.sample_format();
    let rate = supported.sample_rate().0 as f64;
    let config: cpal::StreamConfig = supported.into();
    // every clip at the same peak, so loudness doesn't decide which one sounds better
    let peak = x.iter().fold(0f32, |p, v| p.max(v.abs())).max(1e-6);
    let data: Vec<f32> = resample(x, FS, rate).iter().map(|v| v * 0.7 / peak).collect();
    let position = Arc::new(Mutex::new(0usize));
    match format {
        SampleFormat::F32 => output_stream::<f32>(&device, &config, data, position),
        SampleFormat::I16 => output_stream::<i16>(&device, &config, data, position),
        SampleFormat::I32 => output_stream::<i32>(&device, &config, data, position),
        SampleFormat::U16 => output_stream::<u16>(&device, &config, data, position),
        other => Err(format!("unsupported speaker format {other:?}")),
    }
}

/// Keep the returned stream alive for as long as the clip should play.
pub fn play(x: &[f32]) -> Result<Stream, String> {
    let stream = open_output(x)?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

/// Open every device without starting it: no sound is captured or played.
pub fn check_devices() {
    let default = default_input_name();
    println!("default input: {default:?}");
    for name in input_names() {
        let result = find_input(Some(&name)).and_then(|d| {
            let cfg = d.default_input_config().map_err(|e| e.to_string())?;
            let summary = format!("{} Hz, {} ch, {:?}", cfg.sample_rate().0, cfg.channels(), cfg.sample_format());
            open_input(&d, Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicU32::new(0))).map(|_| summary)
        });
        println!("input  {name}: {result:?}");
    }
    println!("output: {:?}", open_output(&vec![0.0; 4800]).map(|_| "opens"));
}

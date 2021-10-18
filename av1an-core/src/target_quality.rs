use crate::{
  chunk::Chunk,
  process_pipe,
  settings::EncodeArgs,
  vmaf::{self, read_weighted_vmaf},
  Encoder,
};
use ffmpeg_next::format::Pixel;
use splines::{Interpolation, Key, Spline};
use std::{cmp, cmp::Ordering, convert::TryInto, fmt::Error, path::Path, process::Stdio};

// TODO: just make it take a reference to a `Project`
pub struct TargetQuality<'a> {
  vmaf_res: &'a str,
  vmaf_filter: Option<&'a str>,
  vmaf_threads: usize,
  model: Option<&'a Path>,
  probing_rate: usize,
  probes: u32,
  target: f64,
  min_q: u32,
  max_q: u32,
  encoder: Encoder,
  pix_format: Pixel,
  temp: String,
  workers: usize,
  video_params: Vec<String>,
  probe_slow: bool,
}

impl<'a> TargetQuality<'a> {
  pub fn new(args: &'a EncodeArgs) -> Self {
    Self {
      vmaf_res: args.vmaf_res.as_str(),
      vmaf_filter: args.vmaf_filter.as_deref(),
      vmaf_threads: args.vmaf_threads.unwrap_or(0),
      model: args.vmaf_path.as_deref(),
      probes: args.probes,
      target: args.target_quality.unwrap(),
      min_q: args.min_q.unwrap(),
      max_q: args.max_q.unwrap(),
      encoder: args.encoder,
      pix_format: args.pix_format.format,
      temp: args.temp.clone(),
      workers: args.workers,
      video_params: args.video_params.clone(),
      probe_slow: args.probe_slow,
      probing_rate: adapt_probing_rate(args.probing_rate as usize),
    }
  }

  fn per_shot_target_quality(&self, chunk: &Chunk) -> u32 {
    let mut vmaf_cq = vec![];
    let frames = chunk.frames;

    let mut q_list = vec![];

    // Make middle probe
    let middle_point = (self.min_q + self.max_q) / 2;
    q_list.push(middle_point);
    let last_q = middle_point;

    let mut bytes = 0;
    let mut score =
      read_weighted_vmaf(self.vmaf_probe(chunk, last_q as usize, &mut bytes), 0.25).unwrap();
    let mut rate = bytes as f64 * 8. * self.probing_rate as f64 / chunk.frames as f64;
    vmaf_cq.push((score, rate, last_q));

    // Initialize search boundary
    let mut vmaf_lower = score;
    let mut vmaf_upper = score;
    let mut vmaf_cq_lower = last_q;
    let mut vmaf_cq_upper = last_q;
    let mut rate_lower = rate;
    let mut rate_upper = rate;

    let target_rate = 6_000_000. / 24.;

    // Branch
    let next_q = if score < self.target && rate < target_rate {
      self.min_q
    } else {
      self.max_q
    };

    q_list.push(next_q);

    // Edge case check
    score = read_weighted_vmaf(self.vmaf_probe(chunk, next_q as usize, &mut bytes), 0.25).unwrap();
    rate = bytes as f64 * 8. * self.probing_rate as f64 / chunk.frames as f64;
    vmaf_cq.push((score, rate, next_q));

    if (next_q == self.min_q && score < self.target && rate < target_rate)
      || (next_q == self.max_q && score > self.target && rate > target_rate)
    {
      log_probes(
        &mut vmaf_cq,
        frames as u32,
        self.probing_rate as u32,
        &chunk.name(),
        next_q,
        score,
        rate,
        if score < self.target {
          Skip::Low
        } else {
          Skip::High
        },
      );
      return next_q;
    }

    // Set boundary
    if score < self.target && rate < target_rate {
      vmaf_lower = score;
      vmaf_cq_lower = next_q;
      rate_lower = rate;
    } else {
      vmaf_upper = score;
      vmaf_cq_upper = next_q;
      rate_upper = rate;
    }

    // VMAF search
    for _ in 0..self.probes - 2 {
      let new_point = weighted_search(
        f64::from(vmaf_cq_lower),
        vmaf_lower,
        f64::from(vmaf_cq_upper),
        vmaf_upper,
        self.target,
      );

      let new_rate = weighted_rate_search(
        f64::from(vmaf_cq_lower),
        rate_lower,
        f64::from(vmaf_cq_upper),
        rate_upper,
        target_rate,
      );

      let new_point = new_point.max(new_rate);

      if vmaf_cq
        .iter()
        .map(|(_, _, x)| *x)
        .any(|x| x == new_point as u32)
      {
        break;
      }

      q_list.push(new_point as u32);
      score = read_weighted_vmaf(self.vmaf_probe(chunk, new_point, &mut bytes), 0.25).unwrap();
      rate = bytes as f64 * 8. * self.probing_rate as f64 / chunk.frames as f64;
      vmaf_cq.push((score, rate, new_point as u32));

      // Update boundary
      if score < self.target {
        vmaf_lower = score;
        vmaf_cq_lower = new_point as u32;
        rate_lower = rate;
      } else {
        vmaf_upper = score;
        vmaf_cq_upper = new_point as u32;
        rate_upper = rate;
      }
    }

    let (q, q_vmaf, q_rate) = interpolated_target_q(vmaf_cq.clone(), self.target, target_rate);
    log_probes(
      &mut vmaf_cq,
      frames as u32,
      self.probing_rate as u32,
      &chunk.name(),
      q.round() as u32,
      q_vmaf,
      q_rate,
      Skip::None,
    );

    q.round() as u32
  }

  fn vmaf_probe(&self, chunk: &Chunk, q: usize, bytes: &mut u64) -> String {
    let vmaf_threads = if self.vmaf_threads == 0 {
      vmaf_auto_threads(self.workers)
    } else {
      self.vmaf_threads
    };

    let cmd = self.encoder.probe_cmd(
      self.temp.clone(),
      &chunk.name(),
      q,
      self.pix_format,
      self.probing_rate,
      vmaf_threads,
      self.video_params.clone(),
      self.probe_slow,
    );

    let future = async {
      let mut source = if let [pipe_cmd, args @ ..] = &*chunk.source {
        tokio::process::Command::new(pipe_cmd)
          .args(args)
          .stderr(Stdio::piped())
          .stdout(Stdio::piped())
          .spawn()
          .unwrap()
      } else {
        unreachable!()
      };

      let source_pipe_stdout: Stdio = source.stdout.take().unwrap().try_into().unwrap();

      let mut ffmpeg_pipe = if let [ffmpeg, args @ ..] = &*cmd.0 {
        tokio::process::Command::new(ffmpeg)
          .args(args)
          .stdin(source_pipe_stdout)
          .stdout(Stdio::piped())
          .stderr(Stdio::piped())
          .spawn()
          .unwrap()
      } else {
        unreachable!()
      };

      let ffmpeg_pipe_stdout: Stdio = ffmpeg_pipe.stdout.take().unwrap().try_into().unwrap();

      let pipe = if let [cmd, args @ ..] = &*cmd.1 {
        tokio::process::Command::new(cmd.as_ref())
          .args(args.iter().map(AsRef::as_ref))
          .stdin(ffmpeg_pipe_stdout)
          .stdout(Stdio::piped())
          .stderr(Stdio::piped())
          .spawn()
          .unwrap()
      } else {
        unreachable!()
      };

      process_pipe(pipe, chunk.index).await.unwrap();
    };

    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_io()
      .build()
      .unwrap();

    rt.block_on(future);

    let probe_name =
      Path::new(&chunk.temp)
        .join("split")
        .join(format!("v_{}{}.ivf", q, chunk.name()));
    let fl_path = Path::new(&chunk.temp)
      .join("split")
      .join(format!("{}.json", chunk.name()));

    let fl_path = fl_path.to_str().unwrap().to_owned();

    vmaf::run_vmaf(
      &probe_name,
      chunk.source.as_slice(),
      &fl_path,
      self.model.as_ref(),
      self.vmaf_res,
      self.probing_rate,
      self.vmaf_filter,
      self.vmaf_threads,
    )
    .unwrap();

    *bytes = probe_name.metadata().unwrap().len();

    fl_path
  }

  pub fn per_shot_target_quality_routine(&self, chunk: &mut Chunk) {
    chunk.per_shot_target_quality_cq = Some(self.per_shot_target_quality(chunk));
  }
}

pub fn weighted_search(num1: f64, vmaf1: f64, num2: f64, vmaf2: f64, target: f64) -> usize {
  let dif1 = (transform_vmaf(target as f64) - transform_vmaf(vmaf2)).abs();
  let dif2 = (transform_vmaf(target as f64) - transform_vmaf(vmaf1)).abs();

  let tot = dif1 + dif2;

  num1.mul_add(dif1 / tot, num2 * (dif2 / tot)).round() as usize
}

pub fn weighted_rate_search(num1: f64, rate1: f64, num2: f64, rate2: f64, target: f64) -> usize {
  let dif1 = target.ln() - rate2.ln();
  let dif2 = rate1.ln() - target.ln();

  let tot = dif1 + dif2;

  num1.mul_add(dif1 / tot, num2 * (dif2 / tot)).round() as usize
}

pub fn transform_vmaf(vmaf: f64) -> f64 {
  let x: f64 = 1.0 - vmaf / 100.0;
  if vmaf < 99.99 {
    -x.ln()
  } else {
    9.2
  }
}

/// Returns auto detected amount of threads used for vmaf calculation
pub fn vmaf_auto_threads(workers: usize) -> usize {
  const OVER_PROVISION_FACTOR: f64 = 1.25;

  // Logical CPUs
  let threads = num_cpus::get();

  cmp::max(
    ((threads / workers) as f64 * OVER_PROVISION_FACTOR) as usize,
    1,
  )
}

/// Use linear interpolation to get q/crf values closest to the target value
pub fn interpolate_target_q(
  scores: Vec<(f64, f64, u32)>,
  target_vmaf: f64,
  target_rate: f64,
) -> Result<f64, Error> {
  let mut sorted = scores;
  sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

  let keys = sorted
    .iter()
    .map(|(x, _, y)| Key::new(transform_vmaf(*x), f64::from(*y), Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);
  let q_vmaf = spline.clamped_sample(transform_vmaf(target_vmaf)).unwrap();

  sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

  let keys = sorted
    .iter()
    .map(|(_, x, y)| Key::new((*x).ln(), f64::from(*y), Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);
  let q_rate = spline.clamped_sample(target_rate.ln()).unwrap_or(q_vmaf);

  Ok(q_vmaf.max(q_rate))
}

/// Use linear interpolation to get vmaf value that expected from q
pub fn interpolate_target_vmaf(scores: Vec<(f64, f64, u32)>, q: f64) -> Result<f64, Error> {
  let mut sorted = scores;
  sorted.sort_by(|(a, _, _), (b, _, _)| a.partial_cmp(b).unwrap_or(Ordering::Less));

  let keys = sorted
    .iter()
    .map(|f| Key::new(f64::from(f.2), f.0 as f64, Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);

  Ok(spline.clamped_sample(q).unwrap())
}

pub fn interpolate_target_rate(scores: Vec<(f64, f64, u32)>, q: f64) -> Result<f64, Error> {
  let mut sorted = scores;
  sorted.sort_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap_or(Ordering::Less));

  let keys = sorted
    .iter()
    .map(|f| Key::new(f64::from(f.2), f.1 as f64, Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);

  Ok(spline.clamped_sample(q).unwrap())
}

#[derive(Copy, Clone)]
pub enum Skip {
  High,
  Low,
  None,
}

pub fn log_probes(
  vmaf_cq_scores: &mut [(f64, f64, u32)],
  frames: u32,
  probing_rate: u32,
  name: &str,
  target_q: u32,
  target_vmaf: f64,
  target_rate: f64,
  skip: Skip,
) {
  vmaf_cq_scores.sort_by_key(|(_score, _rate, q)| *q);

  info!("Chunk: {}, Rate: {}, Fr {}", name, probing_rate, frames);
  info!(
    "Probes {:?}{}",
    vmaf_cq_scores,
    match skip {
      Skip::High => " Early Skip High Q",
      Skip::Low => " Early Skip Low Q",
      Skip::None => "",
    }
  );
  info!(
    "Target Q: {:.0} VMAF: {:.2} Rate: {:.0}",
    target_q, target_vmaf, target_rate
  );
}

pub const fn adapt_probing_rate(rate: usize) -> usize {
  match rate {
    1..=4 => rate,
    _ => 4,
  }
}

pub fn interpolated_target_q(
  scores: Vec<(f64, f64, u32)>,
  target_vmaf: f64,
  target_rate: f64,
) -> (f64, f64, f64) {
  let q = interpolate_target_q(scores.clone(), target_vmaf, target_rate).unwrap();

  let vmaf = interpolate_target_vmaf(scores.clone(), q).unwrap();
  let rate = interpolate_target_rate(scores, q).unwrap();

  (q, vmaf, rate)
}

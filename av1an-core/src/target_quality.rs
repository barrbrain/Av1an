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

    let mut score = read_weighted_vmaf(self.vmaf_probe(chunk, last_q as usize), 0.25).unwrap();
    vmaf_cq.push((score, last_q));

    // Branch
    let next_q = if score < self.target {
      self.min_q
    } else {
      self.max_q
    };

    q_list.push(next_q);

    // Edge case check
    score = read_weighted_vmaf(self.vmaf_probe(chunk, next_q as usize), 0.25).unwrap();
    vmaf_cq.push((score, next_q));

    if (next_q == self.min_q && score < self.target)
      || (next_q == self.max_q && score > self.target)
    {
      log_probes(
        &mut vmaf_cq,
        frames as u32,
        self.probing_rate as u32,
        &chunk.name(),
        next_q,
        score,
        if score < self.target {
          Skip::Low
        } else {
          Skip::High
        },
      );
      return next_q;
    }

    // VMAF search
    for _ in 0..self.probes - 2 {
      let new_point = weighted_search(&vmaf_cq, self.target).0;

      if vmaf_cq
        .iter()
        .map(|(_, x)| *x)
        .any(|x| x == new_point as u32)
      {
        break;
      }

      q_list.push(new_point as u32);
      score = read_weighted_vmaf(self.vmaf_probe(chunk, new_point as usize), 0.25).unwrap();
      vmaf_cq.push((score, new_point as u32));
    }

    let (q, q_vmaf) = interpolated_target_q(vmaf_cq.clone(), self.target);
    log_probes(
      &mut vmaf_cq,
      frames as u32,
      self.probing_rate as u32,
      &chunk.name(),
      q as u32,
      q_vmaf,
      Skip::None,
    );

    q as u32
  }

  fn vmaf_probe(&self, chunk: &Chunk, q: usize) -> String {
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

    fl_path
  }

  pub fn per_shot_target_quality_routine(&self, chunk: &mut Chunk) {
    chunk.per_shot_target_quality_cq = Some(self.per_shot_target_quality(chunk));
  }
}

pub fn weighted_search(vmaf_cq_scores: &[(f64, u32)], target: f64) -> (u32, f64) {
  let points = vmaf_cq_scores
    .iter()
    .map(|(vmaf, cq)| (*cq, transform_vmaf(*vmaf)))
    .collect::<Vec<_>>();
  let p = lagrange_bisect(&points, transform_vmaf(target));
  (p.0, inverse_transform_vmaf(p.1))
}

pub fn transform_vmaf(vmaf: f64) -> f64 {
  let x: f64 = 1.0 - vmaf / 100.0;
  if vmaf < 99.99 {
    -x.ln()
  } else {
    9.2
  }
}

pub fn inverse_transform_vmaf(y: f64) -> f64 {
  let x: f64 = (-y).exp();
  100.0 * (1.0 - x)
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
pub fn interpolate_target_q(scores: Vec<(f64, u32)>, target: f64) -> Result<f64, Error> {
  let mut sorted = scores;
  sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

  let keys = sorted
    .iter()
    .map(|(x, y)| Key::new(*x, f64::from(*y), Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);

  Ok(spline.sample(target).unwrap())
}

/// Use linear interpolation to get vmaf value that expected from q
pub fn interpolate_target_vmaf(scores: Vec<(f64, u32)>, q: f64) -> Result<f64, Error> {
  let mut sorted = scores;
  sorted.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Less));

  let keys = sorted
    .iter()
    .map(|f| Key::new(f64::from(f.1), f.0 as f64, Interpolation::Linear))
    .collect();

  let spline = Spline::from_vec(keys);

  Ok(spline.sample(q).unwrap())
}

#[derive(Copy, Clone)]
pub enum Skip {
  High,
  Low,
  None,
}

pub fn log_probes(
  vmaf_cq_scores: &mut [(f64, u32)],
  frames: u32,
  probing_rate: u32,
  name: &str,
  target_q: u32,
  target_vmaf: f64,
  skip: Skip,
) {
  vmaf_cq_scores.sort_by_key(|(_score, q)| *q);

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
  info!("Target Q: {:.0} VMAF: {:.2}", target_q, target_vmaf);
}

pub const fn adapt_probing_rate(rate: usize) -> usize {
  match rate {
    1..=4 => rate,
    _ => 4,
  }
}

pub fn interpolated_target_q(scores: Vec<(f64, u32)>, target: f64) -> (f64, f64) {
  let q = interpolate_target_q(scores.clone(), target).unwrap();

  let vmaf = interpolate_target_vmaf(scores, q).unwrap();

  (q, vmaf)
}

#[allow(unused)]
fn lagrange_interpolate(p: &[(u32, f64)], x: u32) -> f64 {
  p.iter()
    .map(|(x0, y0)| {
      let mut num = 1;
      let mut den = 1;
      for (x1, _y1) in p {
        if x0 != x1 {
          num *= i64::from(x) - i64::from(*x1);
          den *= i64::from(*x0) - i64::from(*x1);
        }
      }
      y0 * num as f64 / den as f64
    })
    .sum()
}

#[allow(unused)]
fn lagrange_bisect(p: &[(u32, f64)], y: f64) -> (u32, f64) {
  assert!(p.len() >= 2);

  // Re-center the samples at the target value
  let mut sorted = Vec::from(p);
  for v in &mut sorted {
    v.1 -= y;
  }

  // Order samples by distance from target value
  sorted.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());

  // Take the closest point
  let (mut xb, mut yb) = sorted[0];
  // Take the next close point that brackets the root
  let (mut xa, mut ya) = sorted.iter().find(|&&v| v.1 * yb < 0.).unwrap_or(&(xb, yb));

  loop {
    let x0 = (xa + xb + 1) / 2;
    if x0 == xb || x0 == xa {
      break;
    }

    let y0 = lagrange_interpolate(&sorted, x0);
    if ya * y0 < 0. {
      xb = x0;
      yb = y0;
    } else {
      xa = x0;
      ya = y0;
    }
    if ya.abs() < yb.abs() {
      std::mem::swap(&mut xa, &mut xb);
      std::mem::swap(&mut ya, &mut yb);
    }
  }

  (xb, yb + y)
}

#[cfg(test)]
mod tests {
  use crate::target_quality::lagrange_bisect;

  #[test]
  fn test_bisect() {
    let sorted = vec![(0, 0.0), (1, 1.0), (256, 256.0 * 256.0)];

    assert!(lagrange_bisect(&sorted, 0.0).0 == 0);
    assert!(lagrange_bisect(&sorted, 1.0).0 == 1);
    assert!(lagrange_bisect(&sorted, 256.0 * 256.0).0 == 256);

    assert!(lagrange_bisect(&sorted, 8.0).0 == 3);
    assert!(lagrange_bisect(&sorted, 9.0).0 == 3);

    assert!(lagrange_bisect(&sorted, -1.0).0 == 0);
    assert!(lagrange_bisect(&sorted, 2.0 * 256.0 * 256.0).0 == 256);
  }
}

"""KimoLab: Generate G1 reference motion CSVs from text prompts using Kimodo.

Supports single prompts via CLI or multi-prompt sequences from a YAML file.

Usage:
  # Single prompt
  uv run prompt-to-csv --prompt "A person walks forward" --duration 6.0 --output walk.csv

  # Multi-prompt from YAML
  uv run prompt-to-csv --motion-file motions/walk_and_wave.yaml --output walk_wave.csv

YAML format:
  sequences:
    - prompt: "A person walks forward"
      duration: 6.0
    - prompt: "A person waves with their right hand"
      duration: 3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


KIMODO_G1_MODEL = "kimodo-g1-rp"
KIMODO_G1_FPS = 30


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate G1 motion CSV from text prompts using Kimodo."
  )
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
    "--prompt",
    type=str,
    help="Single text prompt describing the motion.",
  )
  group.add_argument(
    "--motion-file",
    type=str,
    help="Path to a YAML file with a sequence of prompts and durations.",
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=6.0,
    help="Duration in seconds for a single prompt (default: 6.0).",
  )
  parser.add_argument(
    "--output",
    type=str,
    default="motion.csv",
    help="Output CSV file path (default: motion.csv).",
  )
  parser.add_argument(
    "--diffusion-steps",
    type=int,
    default=100,
    help="Number of diffusion steps (default: 100).",
  )
  parser.add_argument(
    "--transition-frames",
    type=int,
    default=5,
    help="Number of transition frames between sequences (default: 5).",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Device to use (default: cuda:0).",
  )
  return parser.parse_args()


def load_motion_file(path: str) -> list[dict[str, str | float]]:
  """Load a YAML motion file with sequences of prompts and durations."""
  with open(path) as f:
    data = yaml.safe_load(f)
  sequences = data.get("sequences", [])
  if not sequences:
    raise ValueError(f"No sequences found in {path}")
  for i, seq in enumerate(sequences):
    if "prompt" not in seq:
      raise ValueError(f"Sequence {i} missing 'prompt' key in {path}")
    if "duration" not in seq:
      seq["duration"] = 6.0
  return sequences


def generate_motion_csv(
  prompts: list[str],
  durations: list[float],
  output_path: str,
  diffusion_steps: int = 100,
  transition_frames: int = 5,
  seed: int | None = None,
  device: str = "cuda:0",
) -> Path:
  """Generate a G1 motion CSV from text prompts using Kimodo.

  Args:
    prompts: List of text prompts.
    durations: List of durations in seconds per prompt.
    output_path: Path to save the CSV.
    diffusion_steps: Number of diffusion denoising steps.
    transition_frames: Frames for blending between prompt segments.
    seed: Random seed.
    device: Torch device.

  Returns:
    Path to the saved CSV file.
  """
  try:
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.tools import seed_everything
  except ImportError as e:
    raise ImportError(
      "Kimodo is required for prompt-based motion generation. "
      "Install it with: pip install kimodo@git+https://github.com/nv-tlabs/kimodo.git"
    ) from e

  if seed is not None:
    seed_everything(seed)

  print(f"Loading Kimodo model: {KIMODO_G1_MODEL}")
  model = load_model(KIMODO_G1_MODEL, device=device)
  fps = model.fps
  print(f"Model loaded (fps={fps})")

  # Convert durations to frame counts
  num_frames = [int(d * fps) for d in durations]

  print("Generating motion from prompts:")
  for prompt, nf, dur in zip(prompts, num_frames, durations):
    print(f"  '{prompt}' -> {nf} frames ({dur}s)")

  output = model(
    prompts,
    num_frames,
    num_denoising_steps=diffusion_steps,
    num_samples=1,
    multi_prompt=True,
    num_transition_frames=transition_frames,
    post_processing=False,  # G1 doesn't use post-processing
    return_numpy=False,
  )

  # Convert to MuJoCo qpos CSV format
  converter = MujocoQposConverter(model.skeleton)
  # root_quat_w_first=False -> xyzw format, which is what mjlab's csv_to_npz expects
  qpos = converter.dict_to_qpos(output, device, root_quat_w_first=False, numpy=True)

  # qpos shape: (1, T, 36) for single sample
  if qpos.ndim == 3:
    qpos = qpos[0]

  # Ground the motion: Kimodo often generates the G1 floating in the air.
  # Shift the root Z so the first frame matches the G1's default standing
  # pelvis height (~0.755m in MuJoCo z-up coordinates).
  G1_DEFAULT_PELVIS_Z = 0.755
  z_offset = qpos[0, 2] - G1_DEFAULT_PELVIS_Z
  if abs(z_offset) > 0.1:
    print(f"Grounding motion: shifting root Z by {-z_offset:.3f}m "
          f"(was {qpos[0, 2]:.3f}m, now {G1_DEFAULT_PELVIS_Z:.3f}m)")
    qpos[:, 2] -= z_offset

  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(str(out), qpos, delimiter=",")

  print(f"Saved motion CSV: {out} ({qpos.shape[0]} frames, {qpos.shape[0] / fps:.1f}s)")
  return out


def main():
  args = parse_args()

  if args.prompt:
    prompts = [args.prompt]
    durations = [args.duration]
  else:
    sequences = load_motion_file(args.motion_file)
    prompts = [s["prompt"] for s in sequences]
    durations = [float(s["duration"]) for s in sequences]

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA not available, falling back to CPU.")
    args.device = "cpu"

  generate_motion_csv(
    prompts=prompts,
    durations=durations,
    output_path=args.output,
    diffusion_steps=args.diffusion_steps,
    transition_frames=args.transition_frames,
    seed=args.seed,
    device=args.device,
  )


if __name__ == "__main__":
  main()

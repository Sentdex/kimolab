"""KimoLab: End-to-end text prompt -> Kimodo G1 motion -> NPZ -> RL training.

Usage:
  # Single prompt
  uv run prompt-train --prompt "A person walks forward" --duration 6.0

  # Multi-prompt from YAML
  uv run prompt-train --motion-file motions/walk_and_wave.yaml

  # With training options
  uv run prompt-train --prompt "A person walks forward" --env.scene.num-envs 4096
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scripts.csv_to_npz import MotionLoader, run_sim
from mjlab.scripts.prompt_to_csv import (
  KIMODO_G1_FPS,
  generate_motion_csv,
  load_motion_file,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def prompt_train(
  # -- Kimodo generation args --
  prompt: str | None = None,
  motion_file: str | None = None,
  duration: float = 6.0,
  diffusion_steps: int = 100,
  transition_frames: int = 5,
  seed: int | None = None,
  # -- csv_to_npz args --
  output_fps: float = 50.0,
  # -- Training args --
  num_envs: int = 4096,
  # -- Misc --
  device: str = "cuda:0",
  skip_wandb: bool = True,
  output_dir: str = "./prompt_motions",
):
  """Generate motion from text prompt and train a tracking policy.

  Args:
    prompt: Single text prompt describing the motion.
    motion_file: Path to YAML file with prompt sequences (alternative to prompt).
    duration: Duration in seconds for single prompt (default: 6.0).
    diffusion_steps: Number of Kimodo diffusion steps.
    transition_frames: Blending frames between prompt segments.
    seed: Random seed for reproducibility.
    output_fps: Output FPS for the NPZ (mjlab sim rate, default: 50).
    num_envs: Number of parallel environments for training.
    device: Torch device.
    skip_wandb: If True, save NPZ locally instead of uploading to WandB.
    output_dir: Directory for generated motion files.
  """
  if prompt is None and motion_file is None:
    print("Error: provide either --prompt or --motion-file")
    sys.exit(1)

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # --- Step 1: Generate motion CSV from text prompt ---
  print("=" * 60)
  print("Step 1: Generating G1 motion from text prompt via Kimodo")
  print("=" * 60)

  if prompt:
    prompts = [prompt]
    durations = [duration]
    motion_name = prompt.lower().replace(" ", "_")[:40]
  else:
    sequences = load_motion_file(motion_file)
    prompts = [s["prompt"] for s in sequences]
    durations = [float(s["duration"]) for s in sequences]
    motion_name = Path(motion_file).stem

  csv_path = out_dir / f"{motion_name}.csv"
  generate_motion_csv(
    prompts=prompts,
    durations=durations,
    output_path=str(csv_path),
    diffusion_steps=diffusion_steps,
    transition_frames=transition_frames,
    seed=seed,
    device=device,
  )

  # --- Step 2: Convert CSV to NPZ ---
  print()
  print("=" * 60)
  print("Step 2: Converting CSV to NPZ via MuJoCo forward kinematics")
  print("=" * 60)

  npz_path = out_dir / f"{motion_name}.npz"
  csv_to_npz_local(
    input_file=str(csv_path),
    output_path=str(npz_path),
    input_fps=KIMODO_G1_FPS,
    output_fps=output_fps,
    device=device,
  )

  # --- Step 3: Train ---
  print()
  print("=" * 60)
  print("Step 3: Training tracking policy")
  print("=" * 60)

  if skip_wandb:
    print(f"Using local NPZ: {npz_path}")
    print(f"To train, run:")
    print(f"  uv run train Mjlab-Tracking-Flat-Unitree-G1 \\")
    print(f"    --motion-file {npz_path} \\")
    print(f"    --env.scene.num-envs {num_envs}")
    print()
    print("Or upload to WandB and use --registry-name instead.")
  else:
    print("Uploading to WandB and starting training...")
    upload_to_wandb(str(npz_path), motion_name)
    # TODO: invoke training with the registry name


def csv_to_npz_local(
  input_file: str,
  output_path: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  """Convert a CSV motion file to NPZ format using MuJoCo FK.

  This is a local version of csv_to_npz that saves directly to a file
  instead of uploading to WandB.
  """
  from mjlab.entity import Entity
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA not available, falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ]

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  from tqdm import tqdm

  scene.reset()
  print(f"Running FK simulation: {motion.output_frames} frames at {output_fps} fps")

  for frame_idx in tqdm(range(motion.output_frames), desc="Processing frames"):
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

    if reset_flag:
      break

  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  np.savez(output_path, **log)
  print(f"Saved NPZ: {output_path}")
  return output_path


def upload_to_wandb(npz_path: str, motion_name: str):
  """Upload NPZ to WandB registry."""
  import wandb

  run = wandb.init(project="kimolab", name=motion_name)
  logged_artifact = run.log_artifact(
    artifact_or_path=npz_path, name=motion_name, type="motions"
  )
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"wandb-registry-motions/{motion_name}",
  )
  print(f"Uploaded to WandB registry: motions/{motion_name}")
  wandb.finish()


def main():
  tyro.cli(prompt_train, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()

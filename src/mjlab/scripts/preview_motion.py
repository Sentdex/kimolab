"""KimoLab: Preview a reference motion NPZ in the native MuJoCo viewer.

Plays the kinematic reference motion through MuJoCo without any RL policy.
Use this to verify your motion looks right before training.

Usage:
  uv run preview-motion prompt_motions/roll_only.npz
  uv run preview-motion prompt_motions/walk_forward.npz --loop
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Preview a reference motion NPZ in the viewer."
  )
  parser.add_argument(
    "npz_file",
    type=str,
    help="Path to the motion NPZ file.",
  )
  parser.add_argument(
    "--loop",
    action="store_true",
    help="Loop the motion continuously.",
  )
  parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (default: 1.0).",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  npz_path = Path(args.npz_file)
  if not npz_path.exists():
    raise FileNotFoundError(f"NPZ file not found: {npz_path}")

  data = np.load(str(npz_path))
  fps = float(data["fps"][0])
  joint_pos = data["joint_pos"]
  body_pos_w = data["body_pos_w"]
  body_quat_w = data["body_quat_w"]
  num_frames = joint_pos.shape[0]
  duration = num_frames / fps

  print(f"Motion: {npz_path.name}")
  print(f"  Frames: {num_frames}")
  print(f"  FPS: {fps}")
  print(f"  Duration: {duration:.1f}s")
  print(f"  Speed: {args.speed}x")
  print()

  # Load the G1 MuJoCo model
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  spec = get_spec()
  # Add a ground plane for reference
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10, 10, 0.01],
    rgba=[0.3, 0.3, 0.3, 1.0],
  )
  mj_model = spec.compile()
  mj_data = mujoco.MjData(mj_model)

  frame_dt = (1.0 / fps) / args.speed

  print(f"Opening viewer... {'Looping' if args.loop else 'Playing once'}.")
  print()

  with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    try:
      while viewer.is_running():
        for frame_idx in range(num_frames):
          if not viewer.is_running():
            break

          # Set root position and orientation
          mj_data.qpos[:3] = body_pos_w[frame_idx, 0]
          mj_data.qpos[3:7] = body_quat_w[frame_idx, 0]
          # Set joint positions
          mj_data.qpos[7:7 + joint_pos.shape[1]] = joint_pos[frame_idx]

          mujoco.mj_forward(mj_model, mj_data)
          viewer.sync()

          time.sleep(frame_dt)

        if not args.loop:
          print("Playback complete. Close viewer window to exit.")
          while viewer.is_running():
            time.sleep(0.1)
          break
    except KeyboardInterrupt:
      print("\nDone.")


if __name__ == "__main__":
  main()

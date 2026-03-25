# KimoLab

**Text-to-physics motion generation.** Type a text prompt, get a physics-trained robot policy.

KimoLab combines [NVIDIA Kimodo](https://huggingface.co/spaces/nvidia/Kimodo) (text-to-motion diffusion) with [mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated robot learning via MuJoCo Warp) to create an end-to-end pipeline:

```
"A person bends down and does a forward somersault."
        |  (Kimodo G1 model)
    G1 joint angles CSV
        |  (MuJoCo forward kinematics)
    NPZ reference motion
        |  (RL training with 4096 parallel envs)
    Physics-based G1 policy
```

## Quick Start

Requires an NVIDIA GPU with CUDA support.

**1. Clone and install:**

```bash
git clone https://github.com/Sentdex/kimolab.git && cd kimolab
git checkout kimolab
uv sync --extra kimodo
```

You also need access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on Hugging Face (accept the license, set `HF_TOKEN`).

### Option A: One Command (prompt straight to training)

```bash
uv run prompt-train \
  --prompt "A person bends down and does a forward somersault." \
  --duration 5.0 --seed 55
```

This generates the motion, converts it, and starts training automatically. All settings are configurable:

```bash
uv run prompt-train \
  --prompt "A person walks forward" \
  --duration 6.0 --seed 42 \
  --num-envs 4096 \
  --save-interval 100 \
  --max-iterations 30000
```

### Option B: Step by Step (preview before training)

**Generate a motion:**

```bash
uv run prompt-to-csv \
  --prompt "A person bends down and does a forward somersault." \
  --duration 5.0 --seed 55 --output motion.csv
```

**Convert and preview:**

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file motion.csv --output-name my_motion \
  --input-fps 30 --output-fps 50 --render False

cp /tmp/motion.npz motion.npz
uv run preview-motion motion.npz --loop
```

**Train (once you're happy with the motion):**

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file motion.npz \
  --env.scene.num-envs 4096 \
  --env.episode-length-s 6.0 \
  --env.terminations.anchor-pos.params.threshold 100.0 \
  --env.terminations.anchor-ori.params.threshold 100.0 \
  --env.terminations.ee-body-pos.params.threshold 100.0 \
  --agent.save-interval 100
```

### Watch the trained policy

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --wandb-run-path your-user/mjlab/run-id \
  --no-terminations True
```

## Multi-Prompt Sequences

Chain multiple motions together using a YAML file:

```yaml
# motions/my_sequence.yaml
sequences:
  - prompt: "A person walks forward confidently"
    duration: 6.0
  - prompt: "A person waves with their right hand"
    duration: 3.0
```

```bash
uv run prompt-to-csv --motion-file motions/my_sequence.yaml --output sequence.csv
```

## New Commands

| Command | Description |
|---------|-------------|
| `uv run prompt-to-csv` | Generate G1 motion CSV from text prompts |
| `uv run prompt-train` | End-to-end pipeline (prompt -> CSV -> NPZ -> train) |
| `uv run preview-motion` | Preview reference motion in MuJoCo viewer |

## How It Works

1. **Kimodo** generates kinematic G1 robot motions directly from text using a diffusion model with an LLM2Vec text encoder (Meta-Llama-3-8B-Instruct backbone)
2. **MuJoCo FK** converts the joint angles to full body positions/orientations/velocities
3. **mjlab** trains a physics-based RL policy (PPO) to track the reference motion in simulation with 4096 parallel environments on GPU via MuJoCo Warp
4. The trained policy outputs joint torques that make the G1 physically execute the motion with real dynamics, contacts, and gravity

## Demo Motion

The included demo uses:
- **Prompt:** "A person bends down and does a forward somersault."
- **Duration:** 5.0 seconds
- **Seed:** 55
- **Model:** kimodo-g1-rp

See `motions/demo_somersault.yaml` for the full config.

---

## Original mjlab Features

All original mjlab features are fully retained. See the [mjlab documentation](https://mujocolab.github.io/mjlab/) for:
- Velocity tracking
- Motion imitation from CSV/WandB
- Multi-GPU training
- Distributed training

## License

KimoLab is licensed under the [Apache License, Version 2.0](LICENSE), same as mjlab.

### Third-Party Code

- **mjlab** — Forked from [mujocolab/mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0)
- **Kimodo** — [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) (Apache-2.0), models under NVIDIA Open Model License
- **`src/mjlab/utils/lab_api/`** — Utilities from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)

## Acknowledgments

- [mjlab](https://github.com/mujocolab/mjlab) team for the excellent GPU-accelerated RL framework
- [NVIDIA Kimodo](https://huggingface.co/spaces/nvidia/Kimodo) team for the text-to-motion generation model
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) team for GPU-accelerated physics

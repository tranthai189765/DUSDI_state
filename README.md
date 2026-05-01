## DUSDi: Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning (NeurIPS 2024)

This codebase was modified based on [URLB](https://github.com/rll-research/url_benchmark).

---

## Installation

### 1. Create conda environment

```sh
conda env create -f conda_env.yml
conda activate dusdi
```

### 2. Install PyTorch with CUDA support

```sh
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
```

> Requires CUDA 11.7. For other CUDA versions see [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Install PettingZoo (for particle environment)

```sh
git clone https://github.com/JiahengHu/Pettingzoo-skill.git
cd Pettingzoo-skill
pip install -e .
cd ..
```

### 4. Install gymnasium-robotics (for AntMaze environment)

```sh
pip install gymnasium-robotics
```

---

## Environments

### Particle (default)

No additional setup required.

### DMC — Humanoid, Quadruped, Hopper & Cheetah

Uses `dm_control` with MuJoCo physics. Set the rendering backend before running:

```sh
# Linux (headless server)
export MUJOCO_GL=egl

# Linux (with display)
export MUJOCO_GL=glfw

# macOS
export MUJOCO_GL=glfw
```

On headless servers (no display), `egl` requires the EGL libraries:
```sh
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev
```

### AntMaze

Uses `gymnasium-robotics`. Observation: ant proprioceptive state (27) + achieved\_goal (2) + desired\_goal (2) = **31 dims**. Action space: **8 dims**, already in `[-1, 1]`. Consistent with TIME's `domain=antmaze` (`AntMaze_UMaze-v5` as primal task).

Supported variants: `antmaze_umaze` (700 steps), `antmaze_medium_play`, `antmaze_medium_diverse`, `antmaze_large_play`, `antmaze_large_diverse` (1000 steps each).

### Ant-v5 (gymnasium MuJoCo)

Uses `gymnasium[mujoco]`. Install:

```sh
pip install "gymnasium[mujoco]"
```

Observation: **27 dims** (`exclude_current_positions_from_observation=True`):

| Indices | Description |
|---------|-------------|
| `[0]` | z-position of torso |
| `[1:5]` | quaternion orientation (w, x, y, z) |
| `[5:13]` | 8 joint angles |
| `[13:19]` | 6 body velocities (vx, vy, vz, roll, pitch, yaw) |
| `[19:27]` | 8 joint velocities |

Action space: **8 dims**, in `[-1, 1]`. Episode length: **1000 steps**. Consistent with TIME's `domain=ant`.

---

## Pre-training

### Particle environment

```sh
python pretrain.py agent=dusdi_diayn domain=particle agent.skill_dim=5 env.particle.N=10 exp_nm="test" use_wandb=false use_tb=true
```

### DMC Humanoid (standard dm_control)

```sh
python pretrain.py domain=dmc_humanoid_state use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4³ = 64 skills):
```sh
python pretrain.py domain=dmc_humanoid_state \
  "agent.training_params.dmc_humanoid_state.skill_dim=3" \
  use_wandb=false use_tb=true
```

### DMC Quadruped (standard dm_control, consistent with url_benchmark)

```sh
python pretrain.py domain=dmc_quadruped_state use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4² = 16 skills):
```sh
python pretrain.py domain=dmc_quadruped_state \
  "agent.training_params.dmc_quadruped_state.skill_dim=2" \
  use_wandb=false use_tb=true
```

### DMC Hopper (standard dm_control, task: `hopper hop`)

```sh
python pretrain.py domain=dmc_hopper_state use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4² = 16 skills):
```sh
python pretrain.py domain=dmc_hopper_state \
  "agent.training_params.dmc_hopper_state.skill_dim=2" \
  use_wandb=false use_tb=true
```

### DMC Cheetah (standard dm_control, task: `cheetah run`)

```sh
python pretrain.py domain=dmc_cheetah_state use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4² = 16 skills):
```sh
python pretrain.py domain=dmc_cheetah_state \
  "agent.training_params.dmc_cheetah_state.skill_dim=2" \
  use_wandb=false use_tb=true
```

### AntMaze (gymnasium-robotics, consistent with TIME `domain=antmaze`)

```sh
# Easy — U-shaped maze (700 steps/episode)
python pretrain.py domain=antmaze_umaze use_wandb=false use_tb=true

# Medium — BigMaze, random start+goal (1000 steps/episode)
python pretrain.py domain=antmaze_medium_play use_wandb=false use_tb=true

# Medium — BigMaze, diverse goal only
python pretrain.py domain=antmaze_medium_diverse use_wandb=false use_tb=true

# Hard — HardestMaze, random start+goal
python pretrain.py domain=antmaze_large_play use_wandb=false use_tb=true

# Hard — HardestMaze, diverse goal only
python pretrain.py domain=antmaze_large_diverse use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4³ = 64 skills):
```sh
python pretrain.py domain=antmaze_umaze \
  "agent.training_params.antmaze_umaze.skill_dim=2" \
  use_wandb=false use_tb=true
```

### Ant-v5 (gymnasium MuJoCo, consistent with TIME `domain=ant`)

```sh
python pretrain.py domain=ant_v5 use_wandb=false use_tb=true
```

With custom seed and GPU:
```sh
python pretrain.py domain=ant_v5 seed=1 cuda_id=0 use_wandb=false use_tb=true
```

Custom skill_dim (default 4 → 4² = 16 skills):
```sh
python pretrain.py domain=ant_v5 \
  "agent.training_params.ant_v5.skill_dim=2" \
  use_wandb=false use_tb=true
```

With W&B logging:
```sh
python pretrain.py domain=ant_v5 seed=1 cuda_id=0 use_wandb=true
```

---

## Skill Channel Configuration

Each environment's observation is split into **channels**; each channel gets its own skill discriminator. Total skills = `skill_dim ^ num_channels`.

### Observation partition per environment

| Environment | Obs dim | Channels | Partition | Default `skill_dim` | Total skills |
|---|---|---|---|---|---|
| `dmc_cheetah_state` | 17 | 2 | `[0:8]` joint angles \| `[8:17]` velocities | 4 | **16** |
| `dmc_hopper_state` | 15 | 2 | `[0:6]` joint angles \| `[6:15]` velocities+touch | 4 | **16** |
| `dmc_quadruped_state` | 78 | 2 | `[0:44]` egocentric state \| `[44:78]` dynamics | 4 | **16** |
| `dmc_humanoid_state` | 67 | 3 | `[0:21]` joint angles \| `[21:37]` spatial \| `[37:67]` velocities | 4 | **64** |
| `antmaze_*` | 31 | 3 | `[0:13]` pose \| `[13:27]` velocity \| `[27:31]` goal info | 4 | **64** |
| `ant_v5` | 27 | 2 | `[0:13]` pose \| `[13:27]` velocity | 4 | **16** |

Override `skill_dim` via CLI: `"agent.training_params.<domain>.skill_dim=<value>"`

| `skill_dim` | 2-channel envs | 3-channel envs |
|---|---|---|
| 2 | 4 skills | 8 skills |
| 3 | 9 skills | 27 skills |
| 4 | 16 skills | 64 skills |
| 5 | 25 skills | 125 skills |

### Full hyperparameter table

| Parameter | Humanoid | Quadruped | Hopper | Cheetah | AntMaze | Ant-v5 |
|---|---|---|---|---|---|---|
| `skill_dim` | 4 | 4 | 4 | 4 | 4 | 4 |
| `num_channels` | 3 | 2 | 2 | 2 | 3 | 2 |
| `total_skills` | 64 | 16 | 16 | 16 | 64 | 16 |
| `update_skill_every_step` | 200 | 200 | 200 | 200 | 200 | 200 |
| `init_temperature` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| `step_count_threshold` | 20 | 20 | 20 | 20 | 50 | 20 |
| `critic_type` | mask_unwt | mask_unwt | mask_unwt | mask_unwt | mask_unwt | mask_unwt |
| `nstep` | 1 | 1 | 1 | 1 | 1 | 1 |

Snapshots are saved to:
```
./models/<obs_type>/<domain>/<experiment>/<seed>/
```

Each checkpoint saves:
- `actor_<frame>.pt` — actor weights
- `critic_<frame>.pt` — critic weights
- `discriminator_<frame>.pt` — DIAYN discriminator weights
- `snapshot_<frame>.pt` — full agent object (for inference)
- `cfg_<frame>.yaml` — full config (architecture and hyperparameters)

---

## Downstream Hierarchical Learning

```sh
python train.py domain=particle ds_task=poison_l low_path="seed:2 particle dusdi_diayn test"
```

---

## Monitoring

### TensorBoard

```sh
tensorboard --logdir exp_local
```

### CSV logs

`train.csv` and `eval.csv` are written to the run directory under `exp_local/`.

### Console output

```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```

| Symbol | Meaning |
|--------|---------|
| F | Total environment frames |
| S | Total agent steps |
| E | Total episodes |
| L | Episode length |
| R | Episode return |
| FPS | Training throughput |
| T | Total training time |

---

## Key changes vs. original `conda_env.yml`

- Removed `gymnasium` (conflicts with `gym==0.21.0` used by stable-baselines3)
- Removed `functorch` (built into PyTorch 2.0)
- Replaced `tb-nightly` with stable `tensorboard`
- Fixed `sklearn==0.0` → `scikit-learn`
- Pinned `dm_control==1.0.14` for reproducibility

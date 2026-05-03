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

Each training run creates an isolated directory under `exp_local/` (timestamped by Hydra):

```
exp_local/<date>/<time>_<domain>_<agent>_seed<seed>_<exp_nm>/
    snapshots/
        actor_<frame>.pt
        critic_<frame>.pt
        discriminator_<frame>.pt
        cfg_<frame>.yaml
    train.csv
    eval.csv
```

Re-running with the same config always creates a new timestamped directory — **no overrides**.

Use `exp_nm` to label different configurations so they are easy to distinguish:

```sh
# Default run
python pretrain.py domain=ant_v5 use_wandb=false use_tb=true
# → exp_local/.../120000_ant_v5_dusdi_diayn_seed2_/snapshots/

# Custom label
python pretrain.py domain=ant_v5 exp_nm=skill2 \
  "agent.training_params.ant_v5.skill_dim=2" use_wandb=false use_tb=true
# → exp_local/.../120500_ant_v5_dusdi_diayn_seed2_skill2/snapshots/
```

To find the latest checkpoint for a domain:
```sh
ls -lt exp_local/**/*ant_v5*/snapshots/ | head -20
```

---

## Downstream Hierarchical Learning

The HRL setup uses a **two-level hierarchy**:
- **Low-level** (frozen): pretrained skill actor from `pretrain.py`
- **High-level** (PPO): selects which skill to activate every `low_level_step=50` steps

### Loading checkpoints

Use `low_snapshot_dir` to point directly to the `snapshots/` folder from any `exp_local/` run:

```sh
python train.py domain=<env> ds_task=<task> \
  low_snapshot_dir="<path_to_exp_local_run>/snapshots" \
  snapshot_ts=<frame>
```

Find available checkpoints:
```sh
ls exp_local/2026.05.01/161417_dmc_cheetah_state_dusdi_diayn_seed2_/snapshots/
# actor_100000.pt  critic_100000.pt  discriminator_100000.pt  cfg_100000.yaml
```

### Supported downstream tasks per domain

| Domain | `ds_task` | dm_control task |
|---|---|---|
| `dmc_hopper_state` | `hopper_hop` | hopper / hop (default) |
| `dmc_hopper_state` | `hopper_stand` | hopper / stand |
| `dmc_cheetah_state` | `cheetah_run` | cheetah / run (default) |
| `dmc_quadruped_state` | `quadruped_run` | quadruped / run (default) |
| `dmc_quadruped_state` | `quadruped_walk` | quadruped / walk |
| `dmc_quadruped_state` | `quadruped_stand` | quadruped / stand |
| `dmc_humanoid_state` | `humanoid_run` | humanoid / run (default) |
| `dmc_humanoid_state` | `humanoid_walk` | humanoid / walk |
| `dmc_humanoid_state` | `humanoid_stand` | humanoid / stand |

`ds_task` format: `<domain_base>_<dmc_task>`, e.g. `hopper_stand` → strips prefix → loads `suite.load('hopper', 'stand')`.

### DMC Cheetah — `cheetah_run`

```sh
python train.py domain=dmc_cheetah_state ds_task=cheetah_run \
  low_snapshot_dir="/workspace/DUSDI_state/exp_local/2026.05.01/161417_dmc_cheetah_state_dusdi_diayn_seed2_/snapshots" \
  snapshot_ts=100000 \
  use_wandb=false use_tb=true
```

### DMC Hopper — `hopper_hop` / `hopper_stand`

```sh
# hop (same task as pretraining)
python train.py domain=dmc_hopper_state ds_task=hopper_hop \
  low_snapshot_dir="/workspace/DUSDI_state/exp_local/2026.05.01/161325_dmc_hopper_state_dusdi_diayn_seed2_/snapshots" \
  snapshot_ts=100000 use_wandb=false use_tb=true

# stand (different downstream task)
python train.py domain=dmc_hopper_state ds_task=hopper_stand \
  low_snapshot_dir="/workspace/DUSDI_state/exp_local/2026.05.01/161325_dmc_hopper_state_dusdi_diayn_seed2_/snapshots" \
  snapshot_ts=100000 use_wandb=false use_tb=true
```

### DMC Quadruped / Humanoid

```sh
python train.py domain=dmc_quadruped_state ds_task=quadruped_run \
  low_snapshot_dir="<path>/snapshots" snapshot_ts=<frame> \
  use_wandb=false use_tb=true

python train.py domain=dmc_humanoid_state ds_task=humanoid_stand \
  low_snapshot_dir="<path>/snapshots" snapshot_ts=<frame> \
  use_wandb=false use_tb=true
```

### Particle (legacy path format)

```sh
python train.py domain=particle ds_task=poison_l low_path="seed:2 particle dusdi_diayn test"
```

### HRL architecture

```
High-level policy (PPO, MlpPolicy)
  │  action: MultiDiscrete([skill_dim] * skill_channel)
  │  e.g. cheetah/hopper: MultiDiscrete([4, 4]) → 16 possible skill combos
  │
  ▼  repeated low_level_step=50 env steps
Low-level policy (frozen pretrained actor)
  │  input: concat(obs, skill_one_hot)   # skill_one_hot shape = skill_channel × skill_dim
  │  output: continuous action ∈ [-1, 1]^act_dim
  │
  ▼
Environment (dm_control / gymnasium)
```

- Effective episode length for high-level: `episode_steps / low_level_step = 1000 / 50 = 20` decisions
- Reward: mean over 50 low-level steps → `reward /= low_level_step`

### PPO hyperparameters

These are SB3 `PPO("MlpPolicy", ...)` defaults, with overrides from `train.yaml`:

| Parameter | Value | Source |
|---|---|---|
| `total_timesteps` | **150 000** | `train.yaml` |
| `n_steps` (rollout per env) | **256** | `train.yaml` |
| `n_env` | **4** | `train.yaml` |
| Rollout buffer size | `256 × 4 = 1024` | derived |
| `batch_size` | 64 | SB3 default |
| `n_epochs` | 10 | SB3 default |
| Mini-batches per rollout | `1024 / 64 = 16` | derived |
| Gradient steps per rollout | `16 × 10 = 160` | derived |
| Total rollouts | `150 000 / 1024 ≈ 146` | derived |
| `learning_rate` | 3e-4 | SB3 default |
| `gamma` | 0.99 | SB3 default |
| `gae_lambda` | 0.95 | SB3 default |
| `clip_range` | 0.2 | SB3 default |
| `normalize_advantage` | True | SB3 default |
| `ent_coef` | 0.0 | SB3 default |
| `vf_coef` | 0.5 | SB3 default |
| `max_grad_norm` | 0.5 | SB3 default |
| Hidden dim (pi + vf) | **[64, 64]** | SB3 MlpPolicy default |

To override any parameter via CLI:
```sh
python train.py domain=dmc_cheetah_state ds_task=cheetah_run \
  low_snapshot_dir="<path>/snapshots" snapshot_ts=100000 \
  total_timesteps=500000 n_steps=512 n_env=8 \
  use_wandb=false use_tb=true
```

### Checkpoint reference

| Parameter | Default | Description |
|---|---|---|
| `low_snapshot_dir` | `""` | Absolute path to `snapshots/` dir (new format) |
| `snapshot_ts` | `4000000` | Which checkpoint frame to load (must exist in snapshots/) |
| `low_level_step` | `50` | Low-level env steps per high-level action |

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

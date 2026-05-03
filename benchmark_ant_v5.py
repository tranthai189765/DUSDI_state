#!/usr/bin/env python3
"""
benchmark_ant_v5.py — DUSDi Ant-v5 skill diversity benchmark

Loads a pretrained DUSDi actor checkpoint and rolls out every skill combination.
Metrics collected (matching TIME's metrics.py conventions):
  1. (x, y) torso trajectories per skill  →  skill_trajectories.png
  2. Mean-obs embeddings, cosine-distance matrix  →  cosine_distance_matrix.png
  3. Episode returns  →  returns.csv + printed summary

Usage:
    python benchmark_ant_v5.py \\
        --snapshot_dir /workspace/DUSDI_state/exp_local/.../snapshots \\
        --snapshot_ts  3000000 \\
        --skill_dim    6 \\
        --skill_channel 2 \\
        --n_episodes   5 \\
        --episode_steps 500 \\
        --cuda_id      0 \\
        --out_dir      ./benchmark_results
"""

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).parent))
from agent.diayn_actors import Actor
import utils  # noqa: F401  — registers SquashedNormal / TruncatedNormal used by Actor.forward


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DUSDi Ant-v5 skill benchmark')
    p.add_argument('--snapshot_dir',  required=True,
                   help='Path to the snapshots/ directory from pretraining')
    p.add_argument('--snapshot_ts',   type=int,   default=3000000)
    p.add_argument('--skill_dim',     type=int,   default=6,
                   help='Number of discrete skills per channel')
    p.add_argument('--skill_channel', type=int,   default=2,
                   help='Number of skill channels (obs partition segments)')
    p.add_argument('--obs_dim',       type=int,   default=105,
                   help='Ant-v5 obs dim (105 with cfrc_ext ON)')
    p.add_argument('--action_dim',    type=int,   default=8)
    p.add_argument('--hidden_dim',    type=int,   default=1024)
    p.add_argument('--n_episodes',    type=int,   default=5,
                   help='Number of episodes to roll out per skill')
    p.add_argument('--episode_steps', type=int,   default=500,
                   help='Max env steps per episode')
    p.add_argument('--cuda_id',       type=int,   default=0)
    p.add_argument('--out_dir',       type=str,   default='./benchmark_results')
    return p.parse_args()


# ── Actor helpers ──────────────────────────────────────────────────────────────

def build_actor(obs_dim, skill_dim, skill_channel, action_dim, hidden_dim, device):
    """Reconstruct the DUSDi 'mono' Actor with the same architecture as pretraining."""
    actor_input_dim = obs_dim + skill_dim * skill_channel
    actor = Actor(
        obs_type='states',
        obs_dim=actor_input_dim,
        action_dim=action_dim,
        feature_dim=hidden_dim,   # ignored for states (uses hidden_dim directly)
        hidden_dim=hidden_dim,
        sac=True,
        log_std_bounds=[-10, 2],
        domain='ant_v5',
    )
    return actor.to(device)


def load_actor(snapshot_dir, snapshot_ts, actor, device):
    actor_path = Path(snapshot_dir) / f'actor_{snapshot_ts}.pt'
    print(f'Loading actor: {actor_path}')
    if not actor_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {actor_path}')
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


# ── Skill helpers ─────────────────────────────────────────────────────────────

def make_skill_vec(skill_tuple, skill_dim, skill_channel, device):
    """One-hot encode (ch0_idx, ch1_idx, ...) → flat float tensor."""
    skill = np.zeros((skill_channel, skill_dim), dtype=np.float32)
    for ch, idx in enumerate(skill_tuple):
        skill[ch, idx] = 1.0
    return torch.as_tensor(skill.flatten(), device=device)


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(actor, env, skill_vec, episode_steps, device):
    """
    Run one episode with a fixed skill vector.
    Returns:
        obs_seq  : (T, obs_dim) float32
        xy_seq   : (T, 2)  torso (x, y) from qpos — NOT included in obs
        ep_return: float
    """
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()

    obs_list, xy_list = [], []
    ep_return = 0.0

    for _ in range(episode_steps):
        # x, y from MuJoCo physics (excluded from obs by default)
        xy_list.append(env.unwrapped.data.qpos[:2].copy())
        obs_list.append(obs.copy())

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            inpt = torch.cat([obs_t, skill_vec], dim=-1)  # [obs(105) | skill(12)]
            action = actor(inpt, 0.2).mean.cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        ep_return += float(reward)
        if terminated or truncated:
            break

    return np.array(obs_list), np.array(xy_list), ep_return


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(skill_xy, all_skills, skill_dim, skill_channel, out_path):
    n_skills = len(all_skills)
    cmap = plt.cm.get_cmap('hsv', n_skills)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, skill_tuple in enumerate(all_skills):
        xy = skill_xy[skill_tuple]
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(i), alpha=0.55, linewidth=0.8)
        ax.scatter(xy[0, 0], xy[0, 1], color=cmap(i), s=30, zorder=5, marker='o')

    ax.set_title(
        f'DUSDi Ant-v5 skill trajectories\n'
        f'skill_dim={skill_dim}, skill_channel={skill_channel}, '
        f'total={n_skills} skills'
    )
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_cosine_matrix(embeddings, all_skills, out_path):
    dist_matrix = cosine_distances(embeddings)
    n = len(all_skills)
    mean_dist = float(np.mean(dist_matrix[np.triu_indices(n, k=1)]))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Cosine distance')
    ax.set_title(f'Skill embedding cosine-distance matrix | mean = {mean_dist:.4f}')
    ax.set_xlabel('Skill index')
    ax.set_ylabel('Skill index')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return mean_dist, dist_matrix


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Actor
    actor = build_actor(args.obs_dim, args.skill_dim, args.skill_channel,
                        args.action_dim, args.hidden_dim, device)
    actor = load_actor(args.snapshot_dir, args.snapshot_ts, actor, device)

    # Environment
    import gymnasium
    env = gymnasium.make(
        'Ant-v5',
        render_mode=None,
        max_episode_steps=args.episode_steps,
        terminate_when_unhealthy=False,
    )

    # Enumerate all skill combinations: skill_dim^skill_channel
    all_skills = list(product(range(args.skill_dim), repeat=args.skill_channel))
    n_skills = len(all_skills)
    print(f'\nBenchmarking {n_skills} skills '
          f'(skill_dim={args.skill_dim}^skill_channel={args.skill_channel}), '
          f'{args.n_episodes} episodes each\n')

    skill_xy      = {}   # skill_tuple -> (T_total, 2) xy array  (for plotting)
    skill_xy_eps  = {}   # skill_tuple -> list of per-episode (T_ep, 2) arrays (for CSV)
    skill_embed   = {}   # skill_tuple -> mean obs vector (obs_dim,)
    skill_returns = {}   # skill_tuple -> list of episode returns

    for s_idx, skill_tuple in enumerate(all_skills):
        skill_vec = make_skill_vec(skill_tuple, args.skill_dim, args.skill_channel, device)
        ep_obs_all, ep_xy_all, ep_rets = [], [], []

        for _ in range(args.n_episodes):
            obs_seq, xy_seq, ret = rollout(actor, env, skill_vec, args.episode_steps, device)
            ep_obs_all.append(obs_seq)
            ep_xy_all.append(xy_seq)
            ep_rets.append(ret)

        skill_xy[skill_tuple]      = np.concatenate(ep_xy_all, axis=0)
        skill_xy_eps[skill_tuple]  = ep_xy_all
        skill_embed[skill_tuple]   = np.mean(np.concatenate(ep_obs_all, axis=0), axis=0)
        skill_returns[skill_tuple] = ep_rets

        mean_ret = np.mean(ep_rets)
        print(f'  [{s_idx+1:3d}/{n_skills}] skill={skill_tuple}  '
              f'mean_return={mean_ret:8.3f}  '
              f'xy_range x=[{skill_xy[skill_tuple][:,0].min():.1f}, {skill_xy[skill_tuple][:,0].max():.1f}]  '
              f'y=[{skill_xy[skill_tuple][:,1].min():.1f}, {skill_xy[skill_tuple][:,1].max():.1f}]')

    env.close()

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_trajectories(skill_xy, all_skills, args.skill_dim, args.skill_channel,
                      out_dir / 'skill_trajectories.png')

    embeddings = np.stack([skill_embed[s] for s in all_skills])
    mean_dist, dist_matrix = plot_cosine_matrix(
        embeddings, all_skills, out_dir / 'cosine_distance_matrix.png'
    )

    # ── Trajectories CSV ─────────────────────────────────────────────────────
    csv_path = out_dir / 'trajectories_ant_v5.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill_tuple', 'episode', 'step', 'x', 'y'])
        for skill_tuple, eps in skill_xy_eps.items():
            for ep_idx, xy_seq in enumerate(eps):
                for step, (x, y) in enumerate(xy_seq):
                    writer.writerow([str(skill_tuple), ep_idx, step, f'{x:.4f}', f'{y:.4f}'])
    print(f'Saved: {csv_path}')

    # ── Summary ──────────────────────────────────────────────────────────────
    all_returns = [r for rs in skill_returns.values() for r in rs]
    all_xy = np.concatenate(list(skill_xy.values()), axis=0)
    x_range = all_xy[:, 0].max() - all_xy[:, 0].min()
    y_range = all_xy[:, 1].max() - all_xy[:, 1].min()

    print('\n' + '='*55)
    print('DUSDi Ant-v5 Benchmark Summary')
    print('='*55)
    print(f'  Checkpoint          : actor_{args.snapshot_ts}.pt')
    print(f'  Skills              : {n_skills} ({args.skill_dim}^{args.skill_channel})')
    print(f'  Episodes per skill  : {args.n_episodes}')
    print(f'  Episode steps       : {args.episode_steps}')
    print(f'  Mean return         : {np.mean(all_returns):.3f} ± {np.std(all_returns):.3f}')
    print(f'  Max / Min return    : {np.max(all_returns):.3f} / {np.min(all_returns):.3f}')
    print(f'  X coverage          : {x_range:.2f} m')
    print(f'  Y coverage          : {y_range:.2f} m')
    print(f'  Cosine dist (mean)  : {mean_dist:.4f}  (higher = more diverse skills)')
    print(f'  Cosine dist (max)   : {dist_matrix[np.triu_indices(n_skills, k=1)].max():.4f}')
    print('='*55)


if __name__ == '__main__':
    main()

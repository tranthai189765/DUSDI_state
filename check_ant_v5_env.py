"""
Inspect the Ant-v5 observation and action spaces at runtime.

Usage:
    python check_ant_v5_env.py

Prints observation dimension, per-index labels, channel partition boundaries,
and action space details.
"""

import numpy as np

# Partition constants (mirrors partition_utils.py)
_ANT_V5_PROPRIO_END = 13  # channel 0: pose [0:13], channel 1: velocity [13:]

OBS_LABELS = [
    "[0]     z-position of torso",
    "[1]     quaternion w",
    "[2]     quaternion x",
    "[3]     quaternion y",
    "[4]     quaternion z",
    "[5]     hip1 angle",
    "[6]     ankle1 angle",
    "[7]     hip2 angle",
    "[8]     ankle2 angle",
    "[9]     hip3 angle",
    "[10]    hip3 angle",
    "[11]    hip4 angle",
    "[12]    ankle4 angle",
    "[13]    body velocity vx",
    "[14]    body velocity vy",
    "[15]    body velocity vz",
    "[16]    body angular velocity roll",
    "[17]    body angular velocity pitch",
    "[18]    body angular velocity yaw",
    "[19]    hip1 joint velocity",
    "[20]    ankle1 joint velocity",
    "[21]    hip2 joint velocity",
    "[22]    ankle2 joint velocity",
    "[23]    hip3 joint velocity",
    "[24]    ankle3 joint velocity",
    "[25]    hip4 joint velocity",
    "[26]    ankle4 joint velocity",
]


def main():
    try:
        import gymnasium
    except ImportError:
        print("ERROR: gymnasium not installed. Run: pip install 'gymnasium[mujoco]'")
        return

    print("=" * 60)
    print("Ant-v5 Environment Inspection")
    print("=" * 60)

    env = gymnasium.make(
        "Ant-v5",
        render_mode=None,
        max_episode_steps=1000,
        terminate_when_unhealthy=False,
    )

    obs, _ = env.reset(seed=42)
    obs = np.asarray(obs, dtype=np.float32).flatten()
    obs_dim = obs.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high
    act_dim = act_low.shape[0]

    print(f"\nObservation space : Box({obs_dim},)  dtype=float32")
    print(f"Action space      : Box({act_dim},)  range=[{act_low.min():.2f}, {act_high.max():.2f}]")

    print("\n--- Observation dimensions ---")
    labels = OBS_LABELS if len(OBS_LABELS) >= obs_dim else OBS_LABELS + [f"[{i}]    (unlabeled)" for i in range(len(OBS_LABELS), obs_dim)]
    for i in range(obs_dim):
        label = labels[i] if i < len(labels) else f"[{i}]"
        print(f"  {label:40s}  value = {obs[i]:+.4f}")

    print("\n--- DUSDi skill channel partition ---")
    print(f"  channel 0  indices [0 : {_ANT_V5_PROPRIO_END}]  "
          f"({_ANT_V5_PROPRIO_END} dims)  body config (pose + joint angles)")
    print(f"  channel 1  indices [{_ANT_V5_PROPRIO_END} : {obs_dim}]  "
          f"({obs_dim - _ANT_V5_PROPRIO_END} dims)  velocities (body + joint)")
    print(f"\nTotal obs_dim = {obs_dim}  |  split point = {_ANT_V5_PROPRIO_END}")
    print(f"Action dim    = {act_dim}")

    # Take one step to verify step API
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    obs2 = np.asarray(obs2, dtype=np.float32).flatten()
    print(f"\n--- Step sanity check ---")
    print(f"  obs shape after step : {obs2.shape}")
    print(f"  reward               : {reward:.4f}")
    print(f"  terminated           : {terminated}  truncated: {truncated}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

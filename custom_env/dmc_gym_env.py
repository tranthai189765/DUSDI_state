import gym
from gym import spaces
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale


class DMCGymEnv(gym.Env):
    """Gym wrapper for dm_control environments, consistent with url_benchmark.

    Uses standard dm_control tasks (no xyz augmentation):
    - humanoid: suite.load('humanoid', 'run')
    - quadruped: suite.load('quadruped', 'run')
    """

    _TASK_MAP = {
        'dmc_humanoid_state': ('humanoid', 'run'),
        'dmc_quadruped_state': ('quadruped', 'run'),
    }

    def __init__(self, domain_key, max_episode_steps=1000, seed=0):
        domain, task = self._TASK_MAP[domain_key]
        env = suite.load(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
        )
        self._dm_env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        ts = self._dm_env.reset()
        obs = ts.observation['observations'].astype(np.float32)
        obs_dim = obs.shape[0]

        action_spec = self._dm_env.action_spec()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32,
        )

    def reset(self):
        self._step_count = 0
        ts = self._dm_env.reset()
        return ts.observation['observations'].astype(np.float32)

    def step(self, action):
        ts = self._dm_env.step(action)
        obs = ts.observation['observations'].astype(np.float32)
        reward = float(ts.reward or 0.0)
        self._step_count += 1
        done = ts.last() or self._step_count >= self._max_episode_steps
        return obs, reward, done, {}

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        return self._dm_env.physics.render(height=64, width=64, camera_id=0)

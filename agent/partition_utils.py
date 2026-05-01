import torch
import numpy as np

from env.env_list import _ANTMAZE_ENVS, _ANT_V5_ENVS

# these values will be initialized upon agent creation, in utils.py
SIMP_PAR = None
USE_IMG = None
DMC_OBS_DIM = None
DMC_ACTION_DIM = None
ANT_V5_OBS_DIM = None   # set dynamically from actual env obs spec
ANT_V5_ACTION_DIM = None
# Ant-v5 obs layout: z(1)+quat(4)+joints(8) = 13 (pose) | body_vel(6)+joint_vel(8) = 14 (velocity)
_ANT_V5_PROPRIO_END = 13  # split point: channel 0 = [0:13], channel 1 = [13:obs_dim]
# AntMaze obs layout: pose(13) | velocity(14) | achieved_goal(2) | desired_goal(2) = 31
# Only the first 27 dims (proprioceptive) are used for skill learning — goals are ignored.
# The pose/velocity split reuses _ANT_V5_PROPRIO_END = 13.
_ANTMAZE_OBS_END = 27  # end of proprioceptive obs; dims [27:31] (goals) are discarded

_ALL_STATE_ENVS = [
    'dmc_humanoid_state', 'dmc_quadruped_state',
    'dmc_hopper_state', 'dmc_cheetah_state',
] + _ANTMAZE_ENVS


def get_env_obs_act_dim(domain, env_config):
	if domain == "particle":
		obs_dim = 70
		if env_config.particle.simplify_action_space:
			action_dim = 20
		else:
			action_dim = 50
	elif domain in _ALL_STATE_ENVS:
		obs_dim = DMC_OBS_DIM
		action_dim = DMC_ACTION_DIM
	elif domain in _ANT_V5_ENVS:
		obs_dim = ANT_V5_OBS_DIM
		action_dim = ANT_V5_ACTION_DIM
	else:
		obs_part, _, action_part = get_env_factorization(domain, 0, 0)
		obs_dim = sum(obs_part)
		action_dim = sum(action_part)

	return obs_dim, action_dim

# legacy function, not actually useful (i.e. the factorization can be arbitrary)
def get_env_factorization(domain, skill_dim, skill_channel):
	if "moma2d" in domain:
		obs_partition = [4, 4, 4, 2, 3, 1]  # base, arm, view, base, arm, view
		action_partition = [2, 3, 1]  # base, arm, view
	elif domain == "particle":
		N = skill_channel
		if USE_IMG:
			obs_partition = [1]*N + [2]*N + [2]*N  # No longer have velocity
		else:
			obs_partition = [1]*N + [4]*N + [2]*N  # lm1-3, pos_vel1-3, lm1-3
		if SIMP_PAR:
			action_partition = [2]*N
		else:
			action_partition = [5]*N

	elif domain == "igibson":
		obs_partition = [3, 4, 3, 8, 7, 7, 2, 5, 1]
		action_partition = [2, 2, 3, 3, 1]
	elif domain == "wipe":
		obs_partition = [96]
		action_partition = [17]
	elif domain in _ANTMAZE_ENVS:
		# obs_partition must sum to full obs_dim (31). Split as [body(27) | goal(4)].
		# The discriminator uses observation_filter to strip goals before skill prediction.
		obs_partition = [_ANTMAZE_OBS_END, DMC_OBS_DIM - _ANTMAZE_OBS_END]
		action_partition = [DMC_ACTION_DIM]
	elif domain in _ALL_STATE_ENVS:
		obs_partition = [DMC_OBS_DIM]
		action_partition = [DMC_ACTION_DIM]
	elif domain in _ANT_V5_ENVS:
		# channel 0: pose [0:13], channel 1: velocity [13:obs_dim]
		obs_partition = [_ANT_V5_PROPRIO_END, ANT_V5_OBS_DIM - _ANT_V5_PROPRIO_END]
		action_partition = [ANT_V5_ACTION_DIM]
	else:
		# For other domain, this is not implemented yet
		raise NotImplementedError

	skill_partition = [skill_dim] * skill_channel
	return obs_partition, skill_partition, action_partition



###############################
# Diayn related specification #
###############################

# specifies how the diayn vector should be partitioned
def get_domain_stats(domain, env_config):
	FULLBOX = env_config.igibson.fullbox
	sep_obj = env_config.igibson.sep_obj
	N = env_config.particle.N

	config = env_config[domain]

	if domain == "toy":
		diayn_dim = 2
		state_partition_points = [0, 1, 2]
	elif domain == "igibson":
		assert FULLBOX
		assert not sep_obj
		diayn_dim = 10
		state_partition_points = [0, 3, 7, 10]
	elif domain == "moma2d":
		diayn_dim = 12
		state_partition_points = [0, 4, 8, 12]
	elif domain == "particle":
		diayn_dim = N * 1
		state_partition_points = list(range(0, diayn_dim+1))
	elif domain in _ANTMAZE_ENVS:
		# 2 channels: pose [0:13] | velocity [13:27]; goal dims [27:31] discarded
		diayn_dim = _ANTMAZE_OBS_END  # 27, not 31
		state_partition_points = [0, _ANT_V5_PROPRIO_END, _ANTMAZE_OBS_END]
	elif domain in _ALL_STATE_ENVS:
		diayn_dim = DMC_OBS_DIM
		state_partition_points = [0, DMC_OBS_DIM]
	elif domain in _ANT_V5_ENVS:
		# 2 channels: pose [0:13] | velocity [13:obs_dim]
		diayn_dim = ANT_V5_OBS_DIM
		state_partition_points = [0, _ANT_V5_PROPRIO_END, ANT_V5_OBS_DIM]
	else:
		raise NotImplementedError

	assert state_partition_points[-1] == diayn_dim
	assert state_partition_points[0] == 0

	return diayn_dim, state_partition_points


# specifies how the diayn vector should be extracted from observation
def observation_filter(obs, domain, env_config):
	if obs is None:
		return None

	config = env_config[domain]

	# We always process in batch
	if len(obs.shape) == 1:
		obs = obs.unsqueeze(0)

	if domain == "toy":
		return obs
	elif domain == "igibson":
		idx = np.array(range(10))
		return obs[:, idx]
	elif domain == "wipe":
		idx = np.array(range(*config.diayn_idx))
		return obs[:, idx]
	elif domain == "moma2d":
		idx = np.array(range(12))
		return obs[:, idx]
	elif domain == "particle":
		idx = np.array(range(env_config.particle.N))
		return obs[:, idx]
	elif domain in _ANTMAZE_ENVS:
		return obs[:, :_ANTMAZE_OBS_END]  # discard goal dims [27:31]
	elif domain in _ALL_STATE_ENVS:
		return obs
	elif domain in _ANT_V5_ENVS:
		return obs
	else:
		print("Domain {} not supported".format(domain))
		raise NotImplementedError


# This function is used for partitioning input into different parts
def obtain_partitions(obs, skill, action, domain, skill_dim, skill_channel):
	obs_partition, skill_partition, action_partition = get_env_factorization(domain, skill_dim, skill_channel)
	# Next, partition both the obs and the skill
	skill_list = torch.split(skill, skill_partition, dim=-1)
	obs_list = torch.split(obs, obs_partition, dim=-1)
	action_list = torch.split(action, action_partition, dim=-1)

	return obs_list, skill_list, action_list

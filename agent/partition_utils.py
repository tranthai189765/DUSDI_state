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
# Ant-v5 obs layout: z(1)+quat(4)+joints(8) = 13 (pose) | body_vel(6)+joint_vel(8)+cfrc(78) = 92
_ANT_V5_PROPRIO_END = 13  # split point: channel 0 = [0:13], channel 1 = [13:obs_dim]
# AntMaze flat obs layout (gymnasium-robotics default, cfrc ON):
#   observation(105) = [0:13] pose | [13:105] velocity+cfrc   (identical to Ant-v5)
#   achieved_goal(2) = [105:107]
#   desired_goal(2)  = [107:109]
# 2 channels mirror Ant-v5 — goal [105:109] is high-level obs only, not in skill partition.
_ANTMAZE_OBS_END = 105  # end of the proprio/cfrc block; goal info starts here

# DMC per-env split points (verified from dm_control obs structure)
# cheetah (17): position(8) | velocity(9)
_CHEETAH_POSE_END = 8
# hopper (15): position(6) | velocity(7)+touch(2)
_HOPPER_POSE_END = 6
# humanoid (67): joint_angles(21) | spatial(head+extremities+torso_vert=16) | velocities(com+joints=30)
_HUMANOID_JOINT_END = 21
_HUMANOID_SPATIAL_END = 37   # 21+16
# quadruped (78): egocentric_state(44) | torso_vel+upright+imu+force_torque(34)
_QUADRUPED_EGOCENTRIC_END = 44

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
		# 2 channels matching Ant-v5: pose [0:13] | velocity+cfrc [13:105]
		# Goal info [105:109] lives in the high-level obs only, not partitioned into skills.
		obs_partition = [_ANT_V5_PROPRIO_END, _ANTMAZE_OBS_END - _ANT_V5_PROPRIO_END]
		action_partition = [DMC_ACTION_DIM]
	elif domain == 'dmc_cheetah_state':
		# 2 channels: joint positions [0:8] | velocities [8:17]
		obs_partition = [_CHEETAH_POSE_END, DMC_OBS_DIM - _CHEETAH_POSE_END]
		action_partition = [DMC_ACTION_DIM]
	elif domain == 'dmc_hopper_state':
		# 2 channels: joint positions [0:6] | velocities+touch [6:15]
		obs_partition = [_HOPPER_POSE_END, DMC_OBS_DIM - _HOPPER_POSE_END]
		action_partition = [DMC_ACTION_DIM]
	elif domain == 'dmc_humanoid_state':
		# 3 channels: joint_angles [0:21] | spatial [21:37] | velocities [37:67]
		obs_partition = [_HUMANOID_JOINT_END, _HUMANOID_SPATIAL_END - _HUMANOID_JOINT_END, DMC_OBS_DIM - _HUMANOID_SPATIAL_END]
		action_partition = [DMC_ACTION_DIM]
	elif domain == 'dmc_quadruped_state':
		# 2 channels: egocentric_state [0:44] | dynamics [44:78]
		obs_partition = [_QUADRUPED_EGOCENTRIC_END, DMC_OBS_DIM - _QUADRUPED_EGOCENTRIC_END]
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
		# 2 channels matching Ant-v5: pose [0:13] | velocity+cfrc [13:105]
		# Discriminator only sees the proprio/cfrc block; goal [105:109] is not part of skill.
		diayn_dim = _ANTMAZE_OBS_END  # 105
		state_partition_points = [0, _ANT_V5_PROPRIO_END, _ANTMAZE_OBS_END]
	elif domain == 'dmc_cheetah_state':
		# 2 channels: positions [0:8] | velocities [8:17]
		diayn_dim = DMC_OBS_DIM
		state_partition_points = [0, _CHEETAH_POSE_END, DMC_OBS_DIM]
	elif domain == 'dmc_hopper_state':
		# 2 channels: positions [0:6] | velocities+touch [6:15]
		diayn_dim = DMC_OBS_DIM
		state_partition_points = [0, _HOPPER_POSE_END, DMC_OBS_DIM]
	elif domain == 'dmc_humanoid_state':
		# 3 channels: joint_angles [0:21] | spatial [21:37] | velocities [37:67]
		diayn_dim = DMC_OBS_DIM
		state_partition_points = [0, _HUMANOID_JOINT_END, _HUMANOID_SPATIAL_END, DMC_OBS_DIM]
	elif domain == 'dmc_quadruped_state':
		# 2 channels: egocentric_state [0:44] | dynamics [44:78]
		diayn_dim = DMC_OBS_DIM
		state_partition_points = [0, _QUADRUPED_EGOCENTRIC_END, DMC_OBS_DIM]
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
		return obs[:, :_ANTMAZE_OBS_END]  # proprio+cfrc block [0:105], exclude goal [105:109]
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

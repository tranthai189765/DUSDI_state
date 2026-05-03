"""record supports for each env"""

_DMC_ENVS = ["dmc_humanoid_state", "dmc_quadruped_state", "dmc_hopper_state", "dmc_cheetah_state"]

_ANTMAZE_ENVS = [
    "antmaze_umaze",
    "antmaze_medium_play",
    "antmaze_medium_diverse",
    "antmaze_large_play",
    "antmaze_large_diverse",
    "antmaze_medium_diverse_dense",   # dense reward variant
    "antmaze_large_diverse_dense",    # dense reward variant
]

_ANT_V5_ENVS = ["ant_v5"]

rv_list = ["moma2d", "particle"]
plot_prediction_list = ["toy", "particle", "moma2d", "particle"]
no_video_eval_list = ["toy", "particle", "moma2d", "particle"] + _DMC_ENVS + _ANTMAZE_ENVS + _ANT_V5_ENVS
save_image_eval_list = ["toy", "moma2d"]

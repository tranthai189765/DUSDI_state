"""record supports for each env"""

_DMC_ENVS = ["dmc_humanoid_state", "dmc_quadruped_state", "dmc_hopper_state"]

rv_list = ["moma2d", "particle"]
plot_prediction_list = ["toy", "particle", "moma2d", "particle"]
no_video_eval_list = ["toy", "particle", "moma2d", "particle"] + _DMC_ENVS
save_image_eval_list = ["toy", "moma2d"]

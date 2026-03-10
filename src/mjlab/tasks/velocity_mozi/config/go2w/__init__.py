import gymnasium as gym

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-Go2w-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2wRoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2wPPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-Go2w-Mozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2wRoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2wPPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-Go2w-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2wFlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2wPPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-Go2w-Mozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2wFlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2wPPORunnerCfg",
  },
)

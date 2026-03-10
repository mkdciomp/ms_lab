import gymnasium as gym

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-G1-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-G1-Mozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-G1-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-G1-Mozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
  },
)

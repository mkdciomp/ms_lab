import gymnasium as gym

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-Go2-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Rough-Unitree-Go2-Mozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-Go2-Mozi",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

gym.register(
  id="ms_lab-Velocity-Flat-Unitree-Go2-Nozi-Play",
  entry_point="ms_lab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeGo2PPORunnerCfg",
  },
)

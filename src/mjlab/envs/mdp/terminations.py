"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ms_lab.managers.scene_entity_config import SceneEntityCfg
from ms_lab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from ms_lab.entity import Entity
  from ms_lab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics state."""
  return NanGuard.detect_nans(env.sim.data)

"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

#from ms_lab.ms_physx.entity import Entity
from ms_lab.managers.manager_term_config import RewardTermCfg
from ms_lab.managers.scene_entity_config import SceneEntityCfg
from ms_lab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from ms_lab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()

def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])

def ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)

def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env._backend.get_robot(asset_cfg.name)
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    dof_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_pos
    return torch.sum((torch.abs(dof_pos - default_joint_pos)), dim=1) * (torch.norm(command[:, :2], dim=1) < 0.1)



def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits.clone().to(env.device)
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


class posture:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env._backend.get_robot(cfg.params["asset_cfg"].name)
    default_joint_pos = asset.data.default_joint_pos #torch.randn([4,29],device="cpu")
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.
    asset: Entity = env._backend.get_robot(asset_cfg.name)
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]


    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids].to(env.device)
    #desired_joint_pos = torch.randn([4,29],device=env.device)
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


def electrical_power_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize electrical power consumption of actuators."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  tau = asset.data.actuator_force
  qd = asset.data.joint_vel
  mech = tau * qd
  mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
  return torch.sum(mech_pos, dim=1)

def joint_position_penalty(
          env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
  ) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env._backend.get_robot(asset_cfg.name)
    cmd = torch.linalg.norm(env.command_manager.get_command("twist"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation."""
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

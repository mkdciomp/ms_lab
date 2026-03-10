"""Useful methods for MDP events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from ms_lab.entity import Entity, EntityIndexing
from ms_lab.managers.scene_entity_config import SceneEntityCfg
from ms_lab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)


if TYPE_CHECKING:
  from ms_lab.envs.manager_based_env import ManagerBasedEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  for entity in env._backend.get_all_robots():
    if not isinstance(entity, Entity):
      continue

    default_root_state = entity.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env._backend.get_env_origins()[env_ids]
    entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
    default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
    entity.write_joint_state_to_sim(
      default_joint_pos, default_joint_vel, env_ids=env_ids
    )


def reset_root_state_uniform(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state for floating-base or mocap fixed-base entities.

  For floating-base entities: Resets pose and velocity via write_root_state_to_sim().
  For fixed-base mocap entities: Resets pose only via write_mocap_pose_to_sim().

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    pose_range: Dictionary with keys {"x", "y", "z", "roll", "pitch", "yaw"}.
    velocity_range: Velocity range (only used for floating-base entities).
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env._backend.get_robot(asset_cfg.name)

  # Pose.
  range_list = [
    pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)


  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )


  # Fixed-based entities with mocap=True.

  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()




    positions = (
      root_states[:, 0:3] + pose_samples[:, 0:3] + env._backend.get_env_origins()[env_ids]
    )

    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )

    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  # Floating-base entities.
  default_root_state = asset.data.default_root_state


  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()



  env_origins_backend = env._backend.get_env_origins()

  positions = (
    root_states[:, 0:3] +
    pose_samples[:, 0:3] +
    env_origins_backend[env_ids]
  )




  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )


  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  # Velocities.
  if velocity_range is None:
    velocity_range = {}
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  velocities[:, 3:] = quat_apply_inverse(orientations, velocities[:, 3:])
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env._backend.get_robot(asset_cfg.name)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits.to(env.device)
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

  joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)


  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def push_by_setting_velocity(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  vel_w = asset.data.root_link_vel_w[env_ids]
  quat_w = asset.data.root_link_quat_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  vel_w[:, 3:] = quat_apply_inverse(quat_w, vel_w[:, 3:])
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
  force_range: tuple[float, float],
  torque_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env._backend.get_robot(asset_cfg.name)
  num_bodies = (
    len(asset_cfg.body_ids)
    if isinstance(asset_cfg.body_ids, list)
    else asset.num_bodies
  )
  size = (len(env_ids), num_bodies, 3)
  forces = sample_uniform(*force_range, size=size, device=env.device)
  torques = sample_uniform(*torque_range, size, env.device)
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )


##
# Domain randomization
##

# TODO: https://github.com/mujocolab/ms_lab/issues/38


@dataclass
class FieldSpec:
  """Specification for how to handle a particular field."""

  entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
  use_address: bool = False  # True for fields that need address (q_adr, v_adr)
  default_axes: Optional[List[int]] = None
  valid_axes: Optional[List[int]] = None


FIELD_SPECS = {
  # Dof - uses addresses.
  "dof_armature": FieldSpec("dof", use_address=True),
  "dof_frictionloss": FieldSpec("dof", use_address=True),
  "dof_damping": FieldSpec("dof", use_address=True),
  # Joint - uses IDs directly.
  "jnt_range": FieldSpec("joint"),
  "jnt_stiffness": FieldSpec("joint"),
  # Body - uses IDs directly.
  "body_mass": FieldSpec("body"),
"body_friction": FieldSpec("body"),
  "body_ipos": FieldSpec("body", default_axes=[0, 1, 2]),
  "body_iquat": FieldSpec("body", default_axes=[0, 1, 2, 3]),
  "body_inertia": FieldSpec("body"),
  "body_pos": FieldSpec("body", default_axes=[0, 1, 2]),
  "body_quat": FieldSpec("body", default_axes=[0, 1, 2, 3]),
  # Geom - uses IDs directly.
  "geom_friction": FieldSpec("geom", default_axes=[0], valid_axes=[0, 1, 2]),
  "geom_pos": FieldSpec("geom", default_axes=[0, 1, 2]),
  "geom_quat": FieldSpec("geom", default_axes=[0, 1, 2, 3]),
  "geom_rgba": FieldSpec("geom", default_axes=[0, 1, 2, 3]),
  # Site - uses IDs directly.
  "site_pos": FieldSpec("site", default_axes=[0, 1, 2]),
  "site_quat": FieldSpec("site", default_axes=[0, 1, 2, 3]),
  # Special case - uses address.
  "qpos0": FieldSpec("joint", use_address=True),
}


def randomize_field(
  env: "ManagerBasedEnv",
  env_ids: torch.Tensor | None,
  field: str,
  ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
  operation: Literal["add", "scale", "abs"] = "abs",
  asset_cfg=None,
  axes: Optional[List[int]] = None,
):
  """Unified model randomization function.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize.
    field: Field name (e.g., "geom_friction", "body_mass").
    ranges: Either (min, max) for all axes, or {axis: (min, max)} for specific axes.
    distribution: Distribution type.
    operation: How to apply randomization.
    asset_cfg: Asset configuration.
    axes: Specific axes to randomize (overrides default_axes from field spec).
  """
  if field not in FIELD_SPECS:
    raise ValueError(
      f"Unknown field '{field}'. Supported fields: {list(FIELD_SPECS.keys())}"
    )

  env._backend.random_field(env_ids,
                                 field,
                                 ranges,
                                 distribution,
                                 operation,
                                 asset_cfg,
                                 axes)




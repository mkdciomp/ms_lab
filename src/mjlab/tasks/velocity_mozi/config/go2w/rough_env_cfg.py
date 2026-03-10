from dataclasses import dataclass, replace

from ms_lab.asset_zoo.robots.unitree_go2w_usd.go2w_constants import (
  GO2W_ACTION_SCALE,
  GO2W_ROBOT_CFG,
)

from ms_lab.tasks.velocity_mozi.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from ms_lab.utils.spec_config import ContactSensorCfg


@dataclass
class UnitreeGo2wRoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{leg}_foot_ground_contact",
        geom1=f"{leg}_foot_collision",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for leg in ["FR", "FL", "RR", "RL"]
    ]
    go2w_cfg = replace(GO2W_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": go2w_cfg}

    self.actions.joint_pos.scale = GO2W_ACTION_SCALE
    self.actions.joint_pos.actuator_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
    self.actions.joint_vel.actuator_names = [".*_foot_joint"]
    self.actions.joint_vel.scale = 0.25

    foot_names = ["FR", "FL", "RR", "RL"]
    sensor_names = [f"{name}_foot_ground_contact" for name in foot_names]
    geom_names = [f"{name}_foot_collision" for name in foot_names]

    #self.rewards.air_time.params["sensor_names"] = sensor_names
    self.rewards.pose.params["std"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }
    self.rewards.pose.params["asset_cfg"].joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


    if self.events.body_mass is not None:
      self.events.body_mass.params["asset_cfg"].body_names = ["base"]
    if self.events.foot_friction is not None:
      self.events.foot_friction.params["asset_cfg"].body_names = [f"{name}_foot" for name in foot_names]

    self.viewer.body_name = "trunk"
    self.viewer.distance = 1.5
    self.viewer.elevation = -10.0

    self.curriculum.command_vel = None
    self.events.body_mass = None
    self.events.foot_friction = None
    self.events.push_robot = None
    self.events.base_external_force_torque = None


@dataclass
class UnitreeGo2wRoughEnvCfg_PLAY(UnitreeGo2wRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    self.events.body_mass = None
    self.events.foot_friction = None

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0

    self.curriculum.command_vel = None
    self.commands.twist.ranges.lin_vel_x = (-3.0, 3.0)
    self.commands.twist.ranges.ang_vel_z = (-3.0, 3.0)

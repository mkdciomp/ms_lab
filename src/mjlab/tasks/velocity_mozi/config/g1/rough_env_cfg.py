from dataclasses import dataclass, replace

from ms_lab.asset_zoo.robots.unitree_g1_usd.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from ms_lab.tasks.velocity_mozi.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from ms_lab.utils.spec_config import ContactSensorCfg


@dataclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": g1_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    geom_names = []
    for i in range(1, 8):
      geom_names.append(f"left_foot{i}_collision")
    for i in range(1, 8):
      geom_names.append(f"right_foot{i}_collision")

    if self.events.body_mass is not None:
      self.events.body_mass.params["asset_cfg"].body_names = ['torso_link']
    if self.events.foot_friction is not None:
      self.events.foot_friction.params["asset_cfg"].body_names = ["left_ankle_pitch_link",
                                                                "right_ankle_pitch_link",
                                                                "left_ankle_roll_link",
                                                                "right_ankle_roll_link"]

    self.actions.joint_pos.scale = G1_ACTION_SCALE
    self.actions.joint_vel = None

    #self.rewards.air_time.params["sensor_names"] = sensor_names
    # self.rewards.pose.params["std"] = {
    #   r"^(left|right)_knee_joint$": 0.6,
    #   r"^(left|right)_hip_pitch_joint$": 0.6,
    #   r"^(left|right)_elbow_joint$": 0.6,
    #   r"^(left|right)_shoulder_pitch_joint$": 0.6,
    #   r"^(?!.*(knee_joint|hip_pitch|elbow_joint|shoulder_pitch)).*$": 0.3,
    # }
    self.rewards.pose.params["std"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      # Arms.
      r".*shoulder_pitch.*": 0.35,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.25,
      r".*wrist.*": 0.3,
    }

    self.viewer.body_name = "torso_link"
    self.commands.twist.viz.z_offset = 0.75

    self.curriculum.command_vel = None
    self.events.body_mass = None
    self.events.foot_friction = None
    self.events.push_robot = None
    self.events.base_external_force_torque = None



@dataclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    self.events.body_mass = None
    self.events.foot_friction = None
    self.events.push_robot = None
    self.events.base_external_force_torque = None

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0

from dataclasses import dataclass

from ms_lab.envs.mdp.actions import joint_actions
from ms_lab.envs.mdp.actions import binary_joint_actions
from ms_lab.managers.action_manager import ActionTerm
from ms_lab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
  actuator_names: list[str]
  """List of actuator names or regex expressions that the action will be mapped to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
  offset: float | dict[str, float] = 0.0
  """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
  preserve_order: bool = False
  """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@dataclass(kw_only=True)
class JointPositionActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointPositionAction
  use_default_offset: bool = True

@dataclass(kw_only=True)
class JointVelocityActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointVelocityAction
  use_default_offset: bool = False

@dataclass(kw_only=True)
class JointEffortsActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointEffortsAction
  use_default_offset: bool = False


@dataclass(kw_only=True)
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """

    joint_names: list[str]
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float]
    """The joint command to move to *open* configuration."""
    close_command_expr: dict[str, float]
    """The joint command to move to *close* configuration."""


@dataclass(kw_only=True)
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


@dataclass(kw_only=True)
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint velocity action term.

    See :class:`BinaryJointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction
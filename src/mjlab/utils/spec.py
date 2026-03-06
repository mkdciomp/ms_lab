"""MjSpec utils."""

from typing import Callable

import mujoco
import numpy as np

from mjlab.actuator.actuator import TransmissionType

_TRANSMISSION_TYPE_MAP = {
  TransmissionType.JOINT: mujoco.mjtTrn.mjTRN_JOINT,
  TransmissionType.TENDON: mujoco.mjtTrn.mjTRN_TENDON,
  TransmissionType.SITE: mujoco.mjtTrn.mjTRN_SITE,
}


def auto_wrap_fixed_base_mocap(
  spec_fn: Callable[[], mujoco.MjSpec],
) -> Callable[[], mujoco.MjSpec]:
  """Wraps spec_fn to auto-wrap fixed-base entities in mocap.

  This enables fixed-base entities to be positioned independently per environment.
  Returns original spec unchanged if entity is floating-base or already mocap.

  .. note::
    Mocap wrapping is automatic, but positioning only happens when you call a
    reset event (e.g., reset_root_state_uniform). Without a reset event, all
    fixed-base robots will remain at the world origin.

  See FAQ: "Why are my fixed-base robots all stacked at the origin?"
  """

  def wrapper() -> mujoco.MjSpec:
    original_spec = spec_fn()

    # Check if entity has freejoint (floating-base).
    free_joint = get_free_joint(original_spec)
    if free_joint is not None:
      return original_spec  # Floating-base, no wrapping needed.

    # Check if root body is already mocap.
    root_body = original_spec.bodies[1] if len(original_spec.bodies) > 1 else None
    if root_body and root_body.mocap:
      return original_spec  # Already mocap, no wrapping needed.

    # Extract and delete keyframes before attach (they transfer but we need
    # them on the wrapper spec, not nested in the attached spec).
    keyframes = [
      (np.array(k.qpos), np.array(k.ctrl), k.name) for k in original_spec.keys
    ]
    for k in list(original_spec.keys):
      original_spec.delete(k)

    # Wrap in mocap body.
    wrapper_spec = mujoco.MjSpec()
    mocap_body = wrapper_spec.worldbody.add_body(name="mocap_base", mocap=True)
    frame = mocap_body.add_frame()
    wrapper_spec.attach(child=original_spec, prefix="", frame=frame)

    # Re-add keyframes to wrapper spec.
    for qpos, ctrl, name in keyframes:
      wrapper_spec.add_key(name=name, qpos=qpos.tolist(), ctrl=ctrl.tolist())

    return wrapper_spec

  return wrapper


def get_non_free_joints(spec: mujoco.MjSpec) -> tuple[mujoco.MjsJoint, ...]:
  """Returns all joints except the free joint."""
  joints: list[mujoco.MjsJoint] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_free_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the free joint. None if no free joint exists."""
  joint: mujoco.MjsJoint | None = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint


def disable_collision(geom: mujoco.MjsGeom) -> None:
  """Disables collision for a geom."""
  geom.contype = 0
  geom.conaffinity = 0


def is_joint_limited(jnt: mujoco.MjsJoint) -> bool:
  """Returns True if a joint is limited."""
  match jnt.limited:
    case mujoco.mjtLimited.mjLIMITED_TRUE:
      return True
    case mujoco.mjtLimited.mjLIMITED_AUTO:
      return jnt.range[0] < jnt.range[1]
    case _:
      return False


def create_motor_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  effort_limit: float,
  gear: float = 1.0,
  armature: float = 0.0,
  frictionloss: float = 0.0,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Create a <motor> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_NONE

  actuator.gear[0] = gear
  # Technically redundant to set both but being explicit here.
  actuator.forcelimited = True
  actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit])

  # Set armature and frictionloss.
  if transmission_type == TransmissionType.JOINT:
    spec.joint(joint_name).armature = armature
    spec.joint(joint_name).frictionloss = frictionloss
  elif transmission_type == TransmissionType.TENDON:
    spec.tendon(joint_name).armature = armature
    spec.tendon(joint_name).frictionloss = frictionloss

  return actuator


def create_position_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  stiffness: float,
  damping: float | None = None,
  dampratio: float | None = None,
  effort_limit: float | None = None,
  armature: float = 0.0,
  frictionloss: float = 0.0,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Create a ``<position>`` actuator on the given spec.

  The control input is a setpoint, not necessarily a desired joint
  angle. MuJoCo computes the effort as ``kp * (ctrl - q) - kv * qdot``, where ``kp`` is
  ``stiffness`` and ``kv`` is the damping coefficient. With low gains, this behaves
  more like a soft torque interface than a hard position servo: the policy can output
  setpoints well outside the kinematic joint limits to modulate the applied torque.
  ``ctrllimited`` is set to ``False`` to allow this.

  Damping can be specified in two ways (exactly one must be provided):

  - ``damping``: raw derivative gain (``kv``). Stored directly.
  - ``dampratio``: dimensionless ratio passed to MuJoCo's ``set_to_position``, which
    resolves it to a ``kv`` at compile time using the effective inertia at ``qpos0``. See
    the *Damping ratio* section in the actuator docs for details.

  If neither is provided, ``kv`` defaults to 0 (no damping). This is used internally
  when ``BuiltinPositionActuatorCfg`` resolves ``dampratio`` at the keyframe
  configuration instead of ``qpos0``.

  Args:
    spec: The MjSpec to add the actuator to.
    joint_name: Name of the target joint (also used as actuator name).
    stiffness: Proportional gain (``kp``).
    damping: Derivative gain (``kv``). Mutually exclusive with ``dampratio``.
    dampratio: Dimensionless damping ratio (1.0 = critical). Mutually exclusive with
      ``damping``.
    effort_limit: Symmetric force limit. If ``None``, unlimited.
    armature: Reflected rotor inertia added to the target.
    frictionloss: Dry friction (stiction) on the target.
    transmission_type: Target type (joint, tendon, or site).
  """
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE

  # set_to_position handles gaintype, biastype, gainprm, and biasprm.
  if dampratio is not None:
    actuator.set_to_position(kp=stiffness, dampratio=dampratio)
  else:
    actuator.set_to_position(kp=stiffness, kv=damping if damping is not None else 0.0)

  # Limits.
  actuator.ctrllimited = False
  # No ctrlrange needed.
  if effort_limit is not None:
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  else:
    actuator.forcelimited = False
    # No forcerange needed.

  # Set armature and frictionloss.
  if transmission_type == TransmissionType.JOINT:
    spec.joint(joint_name).armature = armature
    spec.joint(joint_name).frictionloss = frictionloss
  elif transmission_type == TransmissionType.TENDON:
    spec.tendon(joint_name).armature = armature
    spec.tendon(joint_name).frictionloss = frictionloss

  return actuator


def resolve_dampratio_at_keyframe(
  mj_model: mujoco.MjModel,
  keyframe_name: str,
  actuators: dict[str, tuple[float, float]],
) -> None:
  """Resolve dampratio for selected actuators using a keyframe config.

  Evaluates the mass matrix at the named keyframe and computes
  ``kd = dampratio * 2 * sqrt(kp * M_diag[dof])`` for each actuator, writing the result
  into ``actuator_biasprm[i, 2]``.

  Args:
    mj_model: Compiled MjModel (modified in place).
    keyframe_name: Name of the keyframe whose qpos to use.
    actuators: Map from actuator name to ``(stiffness, dampratio)``.
  """
  mj_data = mujoco.MjData(mj_model)
  mj_data.qpos[:] = mj_model.key(keyframe_name).qpos
  mujoco.mj_forward(mj_model, mj_data)

  M = np.zeros((mj_model.nv, mj_model.nv))
  mujoco.mj_fullM(mj_model, M, mj_data.qM)

  for act_name, (kp, ratio) in actuators.items():
    act = mj_model.actuator(act_name)
    dof = mj_model.joint(act.trnid[0]).dofadr[0]
    kd = ratio * 2.0 * np.sqrt(kp * M[dof, dof])
    act.biasprm[2] = -kd


def create_velocity_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  damping: float,
  effort_limit: float | None = None,
  armature: float = 0.0,
  frictionloss: float = 0.0,
  inheritrange: float = 1.0,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Creates a <velocity> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  actuator.inheritrange = inheritrange
  actuator.ctrllimited = True  # Technically redundant but being explicit.
  actuator.gainprm[0] = damping
  actuator.biasprm[2] = -damping

  if effort_limit is not None:
    # Will this throw an error with autolimits=True?
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  else:
    actuator.forcelimited = False

  if transmission_type == TransmissionType.JOINT:
    spec.joint(joint_name).armature = armature
    spec.joint(joint_name).frictionloss = frictionloss
  elif transmission_type == TransmissionType.TENDON:
    spec.tendon(joint_name).armature = armature
    spec.tendon(joint_name).frictionloss = frictionloss

  return actuator


def create_muscle_actuator(
  spec: mujoco.MjSpec,
  target_name: str,
  *,
  length_range: tuple[float, float] = (0.0, 0.0),
  gear: float = 1.0,
  timeconst: tuple[float, float] = (0.01, 0.04),
  tausmooth: float = 0.0,
  range: tuple[float, float] = (0.75, 1.05),
  force: float = -1.0,
  scale: float = 200.0,
  lmin: float = 0.5,
  lmax: float = 1.6,
  vmax: float = 1.5,
  fpmax: float = 1.3,
  fvmax: float = 1.2,
  transmission_type: TransmissionType = TransmissionType.TENDON,
) -> mujoco.MjsActuator:
  """Create a MuJoCo <muscle> actuator with muscle dynamics.

  Muscles use special activation dynamics and force-length-velocity curves.
  They can actuate tendons or joints.
  """
  actuator = spec.add_actuator(name=target_name, target=target_name)

  if transmission_type not in [TransmissionType.JOINT, TransmissionType.TENDON]:
    raise ValueError("Muscle actuators only support JOINT and TENDON transmissions.")
  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_MUSCLE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_MUSCLE
  actuator.biastype = mujoco.mjtBias.mjBIAS_MUSCLE

  actuator.gear[0] = gear
  actuator.dynprm[0:3] = np.array([*timeconst, tausmooth])
  actuator.gainprm[0:9] = np.array(
    [*range, force, scale, lmin, lmax, vmax, fpmax, fvmax]
  )
  actuator.biasprm[:] = actuator.gainprm[:]
  actuator.lengthrange[0:2] = length_range

  # TODO(kevin): Double check this.
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([0.0, 1.0])

  return actuator

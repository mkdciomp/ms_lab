"""Tests for dampratio support in BuiltinPositionActuator."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.5"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Serial chain where reflected inertia depends on configuration.
# Orthogonal joint axes (Y then Z) ensure that rotating j1 changes how
# link2's inertia projects onto j1's axis, making dof_M0 configuration-dependent.
SERIAL_CHAIN_XML = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="5.0"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="5.0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _make_entity(xml, actuator_cfg, joint_pos=None):
  """Create entity with optional InitialStateCfg joint positions."""
  init_state = EntityCfg.InitialStateCfg()
  if joint_pos is not None:
    init_state.joint_pos = joint_pos
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
    init_state=init_state,
  )
  return Entity(cfg)


def _make_entity_and_sim(xml, actuator_cfg, device, joint_pos=None):
  entity = _make_entity(xml, actuator_cfg, joint_pos)
  mj_model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=mj_model, device=device)
  entity.initialize(mj_model, sim.model, sim.data, device)
  return entity, sim, mj_model


# Config validation.


def test_dampratio_and_damping_mutually_exclusive():
  with pytest.raises(ValueError, match="mutually exclusive"):
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*",), stiffness=100.0, damping=10.0, dampratio=1.0
    )


def test_neither_damping_nor_dampratio_raises():
  with pytest.raises(ValueError, match="Either damping or dampratio"):
    BuiltinPositionActuatorCfg(target_names_expr=(".*",), stiffness=100.0)


def test_damping_only_accepted():
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=(".*",), stiffness=100.0, damping=10.0
  )
  assert cfg.damping == 10.0
  assert cfg.dampratio is None


def test_dampratio_only_accepted():
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=(".*",), stiffness=100.0, dampratio=1.0
  )
  assert cfg.dampratio == 1.0
  assert cfg.damping is None


# Model-level resolution.


def test_dampratio_resolved_after_compile():
  """After compile(), dampratio should be resolved to a negative biasprm[2]."""
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=100.0, dampratio=1.0
  )
  entity = _make_entity(ROBOT_XML, cfg)
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    assert mj_model.actuator_biasprm[i, 2] < 0, (
      "dampratio should be resolved to negative damping after compile"
    )


def test_damping_sets_negative_biasprm2():
  """Raw damping should store a negative value in biasprm[2]."""
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=100.0, damping=10.0
  )
  entity = _make_entity(ROBOT_XML, cfg)
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    assert mj_model.actuator_biasprm[i, 2] == -10.0


def test_dampratio_produces_correct_damping_after_resolution(device):
  """After mujoco_warp resolves dampratio, biasprm[2] should be negative."""
  kp = 100.0
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=kp, dampratio=1.0
  )
  _, sim, _ = _make_entity_and_sim(ROBOT_XML, cfg, device)

  for i in range(sim.model.nu):
    resolved = sim.model.actuator_biasprm[0, i, 2].item()
    assert resolved < 0, "dampratio should be resolved to negative damping"


# Keyframe-aware dampratio resolution.


def _mujoco_ref_damping(xml, kp, dampratio_val, qpos=None):
  """Compute expected damping at a given configuration using mj_fullM.

  This is the reference implementation: build model, set qpos, compute
  mass matrix, then apply kd = dampratio * 2 * sqrt(kp * M_diag[dof]).
  """
  spec = mujoco.MjSpec.from_string(xml)
  joint_names = [jnt.name for jnt in spec.joints]
  for jnt_name in joint_names:
    act = spec.add_actuator(name=jnt_name, target=jnt_name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.set_to_position(kp=kp, kv=0.0)
  mj_model = spec.compile()

  mj_data = mujoco.MjData(mj_model)
  if qpos is not None:
    mj_data.qpos[:] = qpos
  mujoco.mj_forward(mj_model, mj_data)

  nv = mj_model.nv
  M = np.zeros((nv, nv))
  mujoco.mj_fullM(mj_model, M, mj_data.qM)

  results = []
  for i in range(mj_model.nu):
    joint_id = mj_model.actuator_trnid[i, 0]
    dof = mj_model.jnt_dofadr[joint_id]
    M_reflected = M[dof, dof]
    kd = dampratio_val * 2.0 * np.sqrt(kp * M_reflected)
    results.append(-kd)
  return results


def test_dampratio_resolved_at_keyframe_not_qpos0():
  """Dampratio should be resolved using the InitialStateCfg configuration,
  not the default qpos0 (all zeros)."""
  kp = 100.0
  dampratio_val = 1.0
  keyframe_pos = {"j1": 1.0, "j2": -0.5}

  # Reference: at qpos0 (zeros).
  ref_at_qpos0 = _mujoco_ref_damping(SERIAL_CHAIN_XML, kp, dampratio_val)

  # Reference: at keyframe config.
  ref_at_keyframe = _mujoco_ref_damping(
    SERIAL_CHAIN_XML, kp, dampratio_val, qpos=[1.0, -0.5]
  )

  # Verify the two configs actually produce different damping.
  assert ref_at_qpos0 != ref_at_keyframe, (
    "Test setup error: damping should differ between qpos0 and keyframe"
  )

  # mjlab with default dampratio_reference="keyframe" should match.
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("j.*",), stiffness=kp, dampratio=dampratio_val
  )
  entity = _make_entity(SERIAL_CHAIN_XML, cfg, joint_pos=keyframe_pos)
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    np.testing.assert_allclose(
      mj_model.actuator_biasprm[i, 2], ref_at_keyframe[i], rtol=1e-10
    )


def test_dampratio_qpos0_reference_matches_mujoco():
  """dampratio_reference='qpos0' should match vanilla MuJoCo resolution."""
  kp = 100.0
  dampratio_val = 1.0
  keyframe_pos = {"j1": 1.0, "j2": -0.5}

  # Reference: MuJoCo resolves dampratio at qpos0.
  spec = mujoco.MjSpec.from_string(SERIAL_CHAIN_XML)
  for jnt in spec.joints:
    act = spec.add_actuator(name=jnt.name, target=jnt.name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.set_to_position(kp=kp, dampratio=dampratio_val)
  ref_model = spec.compile()
  ref_biasprm2 = [ref_model.actuator_biasprm[i, 2] for i in range(ref_model.nu)]

  # mjlab with dampratio_reference="qpos0" should produce same result,
  # regardless of keyframe config.
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("j.*",),
    stiffness=kp,
    dampratio=dampratio_val,
    dampratio_reference="qpos0",
  )
  entity = _make_entity(SERIAL_CHAIN_XML, cfg, joint_pos=keyframe_pos)
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    np.testing.assert_allclose(
      mj_model.actuator_biasprm[i, 2], ref_biasprm2[i], rtol=1e-10
    )


def test_dampratio_unchanged_when_keyframe_matches_qpos0():
  """When InitialStateCfg matches qpos0, resolution should be the same as
  vanilla MuJoCo."""
  kp = 100.0
  dampratio_val = 1.5

  ref = _mujoco_ref_damping(SERIAL_CHAIN_XML, kp, dampratio_val)

  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("j.*",), stiffness=kp, dampratio=dampratio_val
  )
  # InitialStateCfg defaults to {".*": 0.0}, which matches qpos0.
  entity = _make_entity(SERIAL_CHAIN_XML, cfg)
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    np.testing.assert_allclose(mj_model.actuator_biasprm[i, 2], ref[i], rtol=1e-10)


def test_damping_not_affected_by_keyframe():
  """Raw damping (not dampratio) should be unaffected by InitialStateCfg."""
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("j.*",), stiffness=100.0, damping=10.0
  )
  entity = _make_entity(SERIAL_CHAIN_XML, cfg, joint_pos={"j1": 1.0, "j2": -0.5})
  mj_model = entity.compile()

  for i in range(mj_model.nu):
    assert mj_model.actuator_biasprm[i, 2] == -10.0


def test_keyframe_resolution_propagates_to_mujoco_warp(device):
  """Resolved damping at keyframe config should propagate through to
  mujoco_warp's model."""
  kp = 100.0
  dampratio_val = 1.0
  keyframe_pos = {"j1": 1.0, "j2": -0.5}

  ref_at_keyframe = _mujoco_ref_damping(
    SERIAL_CHAIN_XML, kp, dampratio_val, qpos=[1.0, -0.5]
  )

  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("j.*",), stiffness=kp, dampratio=dampratio_val
  )
  _, sim, _ = _make_entity_and_sim(
    SERIAL_CHAIN_XML, cfg, device, joint_pos=keyframe_pos
  )

  for i in range(sim.model.nu):
    resolved = sim.model.actuator_biasprm[0, i, 2].item()
    np.testing.assert_allclose(resolved, ref_at_keyframe[i], rtol=1e-5)


# Behavioral dynamics tests.

PENDULUM_XML = """
<mujoco>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _simulate_overshoot(xml, dampratio, kp, target, device, steps=300):
  """Simulate and return the peak position overshoot."""
  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("hinge",), stiffness=kp, dampratio=dampratio
  )
  entity, sim, _ = _make_entity_and_sim(xml, cfg, device)

  entity.set_joint_position_target(torch.tensor([[target]], device=device))
  entity.write_data_to_sim()

  max_pos = -float("inf")
  for _ in range(steps):
    sim.step()
    pos = sim.data.qpos[0, 0].item()
    max_pos = max(max_pos, pos)
  return max_pos


def test_higher_dampratio_reduces_overshoot(device):
  """Higher dampratio should produce less overshoot than lower dampratio."""
  kp = 100.0
  target = 0.5

  overshoot_low = _simulate_overshoot(PENDULUM_XML, 0.1, kp, target, device)
  overshoot_high = _simulate_overshoot(PENDULUM_XML, 2.0, kp, target, device)

  assert overshoot_low > overshoot_high, (
    f"Low dampratio overshoot ({overshoot_low:.4f}) should exceed "
    f"high dampratio overshoot ({overshoot_high:.4f})"
  )

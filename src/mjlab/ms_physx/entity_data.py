# -*- coding: utf-8 -*-
"""
MoziSim PhysX 实体数据核心模块
功能描述：
    实现实体（Entity）的状态管理、坐标转换、数据读写等核心功能
    提供世界坐标系与局部坐标系的转换、关节/根节点状态的读写接口
    兼容Isaac Sim/PhysX引擎的底层接口调用
    优化：
    1. 懒加载缓存（按需加载，避免全量数据读取）
    2. 减少GPU/CPU设备拷贝开销
    3. 轻量化缓存失效机制
    4. 预分配张量避免重复创建
    修复1：解决Tensor转C++ int类型失败的问题
    修复2：解决slice类型无法计算len()的问题
作者：规范化整理 + 缓存优化 + 类型修复
日期：2026
依赖说明：
    - torch: 张量计算
    - numpy: 数值计算
    - ms_lab: 内部实体索引定义
    - mozisim: PhysX引擎接口
    - isaaclab: 坐标变换工具函数
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Union, Sequence
# 注：如果以下导入报错，请确保对应模块已安装/路径正确
from ms_lab.entity import EntityIndexing
from ms_lab.third_party.isaaclab.isaaclab.utils.math import combine_frame_transforms
from mozisim.physx_engine.articulations import BatchArticulation


# ============================== 工具函数 ==============================
def get_mapping_matrices(original_order, new_order):
    assert len(original_order) == len(new_order), "原始关节列表和新关节列表长度必须相同"
    mapping_indices = [original_order.index(joint) for joint in new_order]
    n = len(original_order)
    M = np.zeros((n, n), dtype=int)
    for i, idx in enumerate(mapping_indices):
        M[idx, i] = 1
    M_inv = np.zeros((n, n), dtype=int)
    for i, idx in enumerate(mapping_indices):
        M_inv[i, idx] = 1
    return M, M_inv


# ============================== 核心类型转换工具函数 ==============================
def convert_to_python_type(data, max_length: int = None) -> Union[int, list[int]]:
    """
    将Tensor/ndarray/slice类型的索引转换为原生Python整数列表，避免C++接口类型错误

    Args:
        data: 可以是Tensor、ndarray、int、list、slice
        max_length: 当data是slice时，需要的最大长度（用于生成具体的索引列表）

    Returns:
        原生Python类型的索引（int/list[int]）
    """
    if data is None:
        # None代表所有索引，返回0到max_length-1的列表
        if max_length is None:
            raise ValueError("当data为None时，必须指定max_length参数")
        return list(range(max_length))
    elif isinstance(data, slice):
        # 将切片转换为具体的整数列表
        if max_length is None:
            raise ValueError("当data为slice时，必须指定max_length参数")
        start = data.start if data.start is not None else 0
        stop = data.stop if data.stop is not None else max_length
        step = data.step if data.step is not None else 1
        return list(range(start, stop, step))
    elif isinstance(data, torch.Tensor):
        # 如果是标量Tensor，转成Python int；否则转成列表
        if data.numel() == 1:
            return int(data.item())
        else:
            return data.cpu().tolist()
    elif isinstance(data, np.ndarray):
        # 如果是标量ndarray，转成Python int；否则转成列表
        if data.size == 1:
            return int(data.item())
        else:
            return data.tolist()
    elif isinstance(data, int):
        return data
    elif isinstance(data, list):
        # 确保列表中的元素都是整数
        return [int(x) for x in data]
    else:
        raise TypeError(f"不支持的索引类型: {type(data)}")


# ============================== 核心数学转换函数 ==============================
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_w = quat[..., 0:1]
    quat_xyz = quat[..., 1:4]
    cross = torch.cross(quat_xyz, vec, dim=-1)
    return 2 * (quat_w * cross + torch.cross(quat_xyz, cross, dim=-1)) + vec


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_conj = torch.cat([quat[..., 0:1], -quat[..., 1:4]], dim=-1)
    return quat_apply(quat_conj, vec)


def compute_velocity_from_cvel(pos: torch.Tensor, subtree_com: torch.Tensor, cvel: torch.Tensor) -> torch.Tensor:
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]
    offset = subtree_com - pos
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    return torch.cat([lin_vel_w, ang_vel_c], dim=-1)


# ============================== 核心数据类 ==============================
@dataclass
class EntityData:
    indexing: EntityIndexing
    batch_artic: BatchArticulation
    device: str

    default_root_state: torch.Tensor
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor
    default_joint_stiffness: torch.Tensor
    default_joint_damping: torch.Tensor

    default_joint_pos_limits: torch.Tensor
    joint_pos_limits: torch.Tensor
    soft_joint_pos_limits: torch.Tensor

    gravity_vec_w: torch.Tensor
    forward_vec_b: torch.Tensor

    is_fixed_base: bool
    is_articulated: bool
    is_actuated: bool

    POS_DIM = 3
    QUAT_DIM = 4
    LIN_VEL_DIM = 3
    ANG_VEL_DIM = 3
    ROOT_POSE_DIM = POS_DIM + QUAT_DIM
    ROOT_VEL_DIM = LIN_VEL_DIM + ANG_VEL_DIM
    ROOT_STATE_DIM = ROOT_POSE_DIM + ROOT_VEL_DIM

    def __post_init__(self):
        self.num_envs = self.default_root_state.shape[0]
        self.num_bodies = len(self.indexing.body_ids)
        self.num_joints = self.batch_artic.num_joint
        self.num_actuators = self.batch_artic.num_dof

        # ========== 优化：懒加载缓存 ==========
        self._cache = {}
        self._cache_timestamps = {}  # 记录每个缓存项的更新时间
        self._step_counter = 0  # 记录当前step，用于判断缓存是否过期

    def _get_cached_item(self, key, loader_func):
        """
        懒加载缓存核心函数：只在需要时加载指定key的缓存项
        """
        # 检查缓存是否有效（当前step未更新过）
        if key not in self._cache or self._cache_timestamps.get(key, -1) < self._step_counter:
            # 调用加载函数获取数据
            data = loader_func()
            self._cache[key] = data
            self._cache_timestamps[key] = self._step_counter
        return self._cache[key]

    def invalidate_cache(self):
        """优化：仅递增step计数器，而非清空缓存"""
        self._step_counter += 1

    def _resolve_env_ids(self, env_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析并转换环境ID为原生Python整数/列表，避免slice类型导致的len()错误
        """
        return convert_to_python_type(env_ids, max_length=self.num_envs)

    def _resolve_joint_ids(self, joint_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析并转换关节ID为原生Python整数/列表
        """
        return convert_to_python_type(joint_ids, max_length=self.num_actuators)

    def _resolve_body_ids(self, body_ids: Optional[Union[torch.Tensor, slice, int, list]]) -> Union[int, list[int]]:
        """
        解析并转换刚体ID为原生Python整数/列表
        """
        return convert_to_python_type(body_ids, max_length=self.num_bodies)

    # ============================== 属性（懒加载优化） ==============================
    @property
    def heading_w(self) -> torch.Tensor:
        forward_w = quat_apply(self.root_link_quat_w, self.forward_vec_b)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def qacc(self) -> torch.Tensor:
        return self._get_cached_item(
            "qacc",
            lambda: torch.as_tensor(self.batch_artic.get_qacc(), dtype=torch.float32, device=self.device)
        )

    @property
    def joint_acc(self) -> torch.Tensor:
        return self.qacc[:, 6:]

    @property
    def ctrl(self) -> torch.Tensor:
        return self._get_cached_item(
            "dof_position_targets",
            lambda: torch.as_tensor(self.batch_artic.get_dof_position_targets(), dtype=torch.float32, device=self.device)
        )

    @property
    def actuator_force(self) -> torch.Tensor:
        params = self._get_cached_item(
            "dof_drive_params",
            lambda: torch.as_tensor(self.batch_artic.get_dof_drive_params().copy(), dtype=torch.float32,
                                 device=self.device)
        )
        stiffness = params[:, :, 0]
        damping = params[:, :, 1]
        dof_pos = self.joint_pos
        dof_vel = self.joint_vel
        ctrl = self.ctrl
        return stiffness * (ctrl - dof_pos) - damping * dof_vel

    @property
    def contact_sensor_data(self) -> dict[str, torch.Tensor]:
        contact_sensor_datas = self._get_cached_item(
            "contact_sensor_datas",
            lambda: self.batch_artic.get_contact_sensor_datas()
        )
        body_names = self.batch_artic.rigid_body_names
        sensordata = {b: torch.zeros([self.num_envs, 2], dtype=torch.float32, device=self.device) for b in body_names}
        for i in range(self.num_envs):
            robot_datas = contact_sensor_datas[i]
            for idx, rd in enumerate(robot_datas):
                if not rd:
                    continue
                for data in rd:
                    if data.forces.size != 0 and "GroundPlane" in data.actor2:
                        sensordata[body_names[idx]][i, 0] = 1
        return sensordata

    @property
    def subtree_com(self) -> torch.Tensor:
        link_c_mass_pose = self._get_cached_item(
            "link_c_mass_pose",
            lambda: self.batch_artic.get_link_c_mass_pose(rigid_body_indices=[0])
        )
        return torch.tensor(link_c_mass_pose[:, :, :3], dtype=torch.float32, device=self.device)

    @property
    def cvel(self) -> torch.Tensor:
        rigid_body_velocity = self._get_cached_item(
            "rigid_body_velocity",
            lambda: self.batch_artic.get_rigid_body_velocity(rigid_body_indices=[0])
        )
        return torch.tensor(rigid_body_velocity, dtype=torch.float32, device=self.device)

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        # 懒加载root_pose
        root_pose = self._get_cached_item(
            "root_pose",
            lambda: torch.as_tensor(self.batch_artic.get_root_pose().copy(), dtype=torch.float32, device=self.device)
        )
        # 预分配张量避免重复创建
        return self._get_cached_item(
            "root_link_pose_w",
            lambda: self._compute_root_link_pose_w(root_pose)
        )

    def _compute_root_link_pose_w(self, root_pose):
        """辅助函数：计算root_link_pose_w"""
        # 创建原张量的视图，避免数据拷贝
        new_pose = root_pose.clone().to(self.device)
        # 直接交换指定列的值
        new_pose[:, 3:7] = new_pose[:, [6, 3, 4, 5]]
        return new_pose

    @property
    def root_link_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        return self.root_link_pose_w[:, 3:]

    @property
    def root_link_vel_w(self) -> torch.Tensor:

        return self._get_cached_item(
            "root_link_vel_w",
            lambda: torch.as_tensor(self.batch_artic.get_root_velocity(), dtype=torch.float32, device=self.device)
        )

    @property
    def root_link_lin_vel_w(self):
        return self.root_link_vel_w[..., :3]

    @property
    def root_link_ang_vel_w(self):
        return self.root_link_vel_w[..., 3:]

    @property
    def root_link_lin_vel_b(self):
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self):
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_forward_vector_w(self):
        return quat_apply(self.root_link_quat_w, self.forward_vec_b)

    @property
    def projected_gravity_b(self):
        return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

    @property
    def joint_pos(self):
        return self._get_cached_item(
            "dof_positions",
            lambda: torch.as_tensor(self.batch_artic.get_dof_positions(), dtype=torch.float32, device=self.device)
        )

    @property
    def joint_vel(self):
        return self._get_cached_item(
            "dof_velocities",
            lambda: torch.as_tensor(self.batch_artic.get_dof_velocities(), dtype=torch.float32, device=self.device)
        )

    def site_pose_w(self, site_name: str, offset: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        rigid_body_names = self.batch_artic.rigid_body_names
        site_id = next((i for i, n in enumerate(rigid_body_names) if n == site_name), -1)
        if self.is_fixed_base and site_id >= 0:
            site_id -= 1

        # 懒加载rigid_body_pose
        rigid_body_pose = self._get_cached_item(
            "rigid_body_pose",
            lambda: self.batch_artic.get_rigid_body_pose()
        )
        site_pose = rigid_body_pose[:, site_id]
        site_pos = site_pose[:3]
        site_quat = np.array([site_pose[6], *site_pose[3:6]])

        ee_offset_pos = offset[:3]
        ee_offset_quat = offset[3:]
        if isinstance(ee_offset_pos, np.ndarray):
            ee_offset_pos = torch.tensor(ee_offset_pos, device=self.device)
        if isinstance(ee_offset_quat, np.ndarray):
            ee_offset_quat = torch.tensor(ee_offset_quat, device=self.device)

        site_pos_t = torch.tensor(site_pos, device=self.device)
        site_quat_t = torch.tensor(site_quat, device=self.device)
        ee_pos, ee_quat = combine_frame_transforms(site_pos_t, site_quat_t, ee_offset_pos, ee_offset_quat)
        return torch.cat([ee_pos, ee_quat], dim=0)

    # ============================== 写入接口（优化拷贝） ==============================
    def write_root_pose(self, pose: torch.Tensor, env_ids=None):
        if self.is_fixed_base:
            raise ValueError("Fixed base cannot write root pose")
        env_ids = self._resolve_env_ids(env_ids)

        # 预分配张量避免重复创建
        new_pose = torch.empty_like(pose, device=self.device)
        new_pose[:, :3] = pose[:, :3]
        new_pose[:, 6] = pose[:, 3]
        new_pose[:, 3:6] = pose[:, 4:]

        # 优化：仅在必要时拷贝到CPU
        if new_pose.device != torch.device("cpu"):
            new_pose_np = new_pose.detach().cpu().numpy()
        else:
            new_pose_np = new_pose.numpy()

        self.batch_artic.set_root_pose(new_pose_np, env_ids)
        self.invalidate_cache()

    def write_root_velocity(self, vel: torch.Tensor, env_ids=None):
        if self.is_fixed_base:
            raise ValueError("Fixed base cannot write root velocity")
        env_ids = self._resolve_env_ids(env_ids)

        # 优化：仅在必要时拷贝到CPU
        if vel.device != torch.device("cpu"):
            vel_np = vel.detach().cpu().numpy()
        else:
            vel_np = vel.numpy()

        self.batch_artic.set_root_velocity(vel_np, env_ids)
        self.invalidate_cache()

    def write_joint_state(self, position, velocity, joint_ids=None, env_ids=None):
        self.write_joint_position(position, joint_ids, env_ids)
        self.write_joint_velocity(velocity, joint_ids, env_ids)

    def write_joint_position(self, position, joint_ids=None, env_ids=None):
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        # 优化：仅在必要时拷贝到CPU
        if position.device != torch.device("cpu"):
            pos_np = position.detach().cpu().numpy()
        else:
            pos_np = position.numpy()

        try:
            self.batch_artic.set_dof_positions(pos_np, env_ids, joint_ids)
        except Exception as e:
            # 增加调试信息，方便定位具体的类型问题
            raise RuntimeError(
                f"设置关节位置失败 - "
                f"position类型: {type(pos_np)}, shape: {pos_np.shape}, "
                f"env_ids类型: {type(env_ids)}, 值: {env_ids}, "
                f"joint_ids类型: {type(joint_ids)}, 值: {joint_ids}"
            ) from e

        self.invalidate_cache()

    def write_joint_velocity(self, velocity, joint_ids=None, env_ids=None):
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        # 优化：仅在必要时拷贝到CPU
        if velocity.device != torch.device("cpu"):
            vel_np = velocity.detach().cpu().numpy()
        else:
            vel_np = velocity.numpy()

        self.batch_artic.set_dof_velocities(vel_np, env_ids, joint_ids)
        self.invalidate_cache()

    def write_joint_efforts(self, efforts, joint_ids=None, env_ids=None):
        if not self.is_articulated:
            raise ValueError("Not articulated")
        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = self._resolve_joint_ids(joint_ids)

        # 优化：仅在必要时拷贝到CPU
        if efforts.device != torch.device("cpu"):
            efforts_np = efforts.detach().cpu().numpy()
        else:
            efforts_np = efforts.numpy()

        self.batch_artic.set_dof_efforts(efforts_np, env_ids, joint_ids)
        self.invalidate_cache()

    def write_external_wrench(self, force=None, torque=None, body_ids=None, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        if force is None:
            force = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        if torque is None:
            torque = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

        data = torch.cat([force, torque], dim=-1)

        # 优化：仅在必要时拷贝到CPU
        if data.device != torch.device("cpu"):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data.numpy()

        self.batch_artic.set_link_force_torque(data_np, env_ids, body_ids)
        self.invalidate_cache()

    def write_ctrl(self, ctrl, ctrl_ids=None, env_ids=None):
        if not self.is_actuated:
            raise ValueError("Not actuated")
        env_ids = self._resolve_env_ids(env_ids)
        ctrl_ids = self._resolve_joint_ids(ctrl_ids)  # ctrl_ids本质是关节ID

        # 优化：仅在必要时拷贝到CPU
        if ctrl.device != torch.device("cpu"):
            ctrl_np = ctrl.detach().cpu().numpy()
        else:
            ctrl_np = ctrl.numpy()

        try:
            self.batch_artic.set_dof_position_targets(ctrl_np, env_ids, ctrl_ids)
        except Exception as e:
            raise RuntimeError(
                f"设置关节位置目标失败 - "
                f"ctrl类型: {type(ctrl_np)}, shape: {ctrl_np.shape}, "
                f"env_ids类型: {type(env_ids)}, 值: {env_ids}, "
                f"ctrl_ids类型: {type(ctrl_ids)}, 值: {ctrl_ids}"
            ) from e
        self.invalidate_cache()

    def write_velocity(self, ctrl, ctrl_ids=None, env_ids=None):
        if not self.is_actuated:
            raise ValueError("Not actuated")
        env_ids = self._resolve_env_ids(env_ids)
        ctrl_ids = self._resolve_joint_ids(ctrl_ids)

        # 优化：仅在必要时拷贝到CPU
        if ctrl.device != torch.device("cpu"):
            ctrl_np = ctrl.detach().cpu().numpy()
        else:
            ctrl_np = ctrl.numpy()

        self.batch_artic.set_dof_velocity_targets(ctrl_np, env_ids, ctrl_ids)
        self.invalidate_cache()

    def set_rigid_body_mass(self, mass, env_ids, body_ids):
        mass = mass.unsqueeze(1)

        # 优化：仅在必要时拷贝到CPU
        if mass.device != torch.device("cpu"):
            mass_np = mass.detach().cpu().numpy()
        else:
            mass_np = mass.numpy()

        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        self.batch_artic.set_rigid_body_mass(mass_np, env_ids, body_ids)
        self.invalidate_cache()

    def set_body_link_friction_params(self, params, env_ids, body_ids):
        sf = params[:, :, 0]
        df = params[:, :, 1]

        # 优化：仅在必要时拷贝到CPU
        if sf.device != torch.device("cpu"):
            sf_np = sf.detach().cpu().numpy()
        else:
            sf_np = sf.numpy()

        if df.device != torch.device("cpu"):
            df_np = df.detach().cpu().numpy()
        else:
            df_np = df.numpy()

        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        self.batch_artic.set_link_static_friction(sf_np, env_ids, body_ids)
        self.batch_artic.set_link_dynamic_friction(df_np, env_ids, body_ids)
        self.invalidate_cache()

    def get_rigid_body_mass(self):
        return self._get_cached_item(
            "rigid_body_mass",
            lambda: torch.as_tensor(self.batch_artic.get_rigid_body_mass(), dtype=torch.float32, device=self.device)
        )

    def get_body_link_friction_params(self):
        return self._get_cached_item(
            "link_friction_params",
            lambda: torch.as_tensor(self.batch_artic.get_link_friction_params(), dtype=torch.float32, device=self.device)
        )


# ============================== 测试 ==============================
if __name__ == "__main__":
    device = "cpu"
    num_envs = 4
    num_bodies = 5
    num_joints = 4


    # 模拟EntityIndexing（实际使用时替换为真实实例）
    class MockEntityIndexing:
        def __init__(self, body_ids, geom_ids, site_ids, ctrl_ids, joint_ids, mocap_id,
                     joint_q_adr, joint_v_adr, free_joint_q_adr, free_joint_v_adr, sensor_adr):
            self.body_ids = body_ids
            self.geom_ids = geom_ids
            self.site_ids = site_ids
            self.ctrl_ids = ctrl_ids
            self.joint_ids = joint_ids
            self.mocap_id = mocap_id
            self.joint_q_adr = joint_q_adr
            self.joint_v_adr = joint_v_adr
            self.free_joint_q_adr = free_joint_q_adr
            self.free_joint_v_adr = free_joint_v_adr
            self.sensor_adr = sensor_adr


    indexing = MockEntityIndexing(
        body_ids=torch.arange(num_bodies, device=device),
        geom_ids=torch.arange(3, device=device),
        site_ids=torch.arange(2, device=device),
        ctrl_ids=torch.arange(4, device=device),
        joint_ids=torch.arange(num_joints, device=device),
        mocap_id=None,
        joint_q_adr=torch.arange(7, 7 + num_joints, device=device),
        joint_v_adr=torch.arange(6, 6 + num_joints, device=device),
        free_joint_q_adr=torch.arange(7, device=device),
        free_joint_v_adr=torch.arange(6, device=device),
        sensor_adr={"force": torch.arange(3, device=device)}
    )


    # 模拟BatchArticulation（实际使用时替换为真实实例）
    class MockBatchArticulation:
        def __init__(self):
            self.num_joint = num_joints
            self.num_dof = num_joints
            self.rigid_body_names = [f"body_{i}" for i in range(num_bodies)]

        # 模拟底层接口（实际使用时替换为真实调用）
        def get_root_pose(self): return np.zeros((num_envs, 7))

        def get_dof_positions(self): return np.zeros((num_envs, num_joints))

        def get_dof_velocities(self): return np.zeros((num_envs, num_joints))

        def get_rigid_body_pose(self): return np.zeros((num_envs, num_bodies, 7))

        def get_rigid_body_velocity(self, rigid_body_indices): return np.zeros((num_envs, num_bodies, 6))

        def get_link_c_mass_pose(self, rigid_body_indices): return np.zeros((num_envs, num_bodies, 7))

        def get_contact_sensor_datas(self): return [[] for _ in range(num_envs)]

        def get_dof_drive_params(self): return np.zeros((num_envs, num_joints, 2))

        def get_qacc(self): return np.zeros((num_envs, 6 + num_joints))

        def get_dof_position_targets(self): return np.zeros((num_envs, num_joints))

        def get_rigid_body_mass(self): return np.zeros((num_envs, num_bodies))

        def get_link_friction_params(self): return np.zeros((num_envs, num_bodies, 2))

        def set_root_pose(self, pose, env_ids): pass

        def set_root_velocity(self, vel, env_ids): pass

        def set_dof_positions(self, pos, env_ids, joint_ids): pass

        def set_dof_velocities(self, vel, env_ids, joint_ids): pass

        def set_dof_efforts(self, efforts, env_ids, joint_ids): pass

        def set_link_force_torque(self, data, env_ids, body_ids): pass

        def set_dof_position_targets(self, ctrl, env_ids, ctrl_ids): pass

        def set_dof_velocity_targets(self, ctrl, env_ids, ctrl_ids): pass

        def set_rigid_body_mass(self, mass, env_ids, body_ids): pass

        def set_link_static_friction(self, sf, env_ids, body_ids): pass

        def set_link_dynamic_friction(self, df, env_ids, body_ids): pass


    batch_artic = MockBatchArticulation()

    # 初始化默认参数
    default_root_state = torch.zeros(num_envs, 13, device=device)
    default_joint_pos = torch.zeros(num_envs, num_joints, device=device)
    default_joint_vel = torch.zeros(num_envs, num_joints, device=device)
    default_joint_stiffness = torch.zeros(num_envs, num_joints, device=device)
    default_joint_damping = torch.zeros(num_envs, num_joints, device=device)
    default_joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    soft_joint_pos_limits = torch.zeros(num_joints, 2, device=device)
    gravity_vec_w = torch.tensor([0.0, 0.0, -9.81], device=device)
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=device)

    # 创建EntityData实例
    entity_data = EntityData(
        indexing=indexing,
        batch_artic=batch_artic,
        device=device,
        default_root_state=default_root_state,
        default_joint_pos=default_joint_pos,
        default_joint_vel=default_joint_vel,
        default_joint_stiffness=default_joint_stiffness,
        default_joint_damping=default_joint_damping,
        default_joint_pos_limits=default_joint_pos_limits,
        joint_pos_limits=joint_pos_limits,
        soft_joint_pos_limits=soft_joint_pos_limits,
        gravity_vec_w=gravity_vec_w,
        forward_vec_b=forward_vec_b,
        is_fixed_base=False,
        is_articulated=True,
        is_actuated=True
    )

    print("✅ EntityData 优化版初始化完成！")
    print(f"  - 环境数量: {entity_data.num_envs}")
    print(f"  - 刚体数量: {entity_data.num_bodies}")
    print(f"  - 关节数量: {entity_data.num_joints}")
    print(f"  - 执行器数量: {entity_data.num_actuators}")

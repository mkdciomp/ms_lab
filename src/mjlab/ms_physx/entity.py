# -*- coding: utf-8 -*-

from __future__ import annotations

# 系统库导入
from operator import setitem
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence, Optional, Union, Tuple
import re
from enum import Enum

# 第三方库导入
import torch
import numpy as np
from six import reraise

# 内部模块导入
from operator import setitem
from ms_lab.ms_physx.entity_data import EntityData
from mozisim.physx_engine.articulations import BatchArticulation
from ms_lab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names

# ============================== 全局配置常量 ==============================
"""
设备配置说明：
    - 自动检测CUDA可用性，优先使用GPU加速
    - 可手动指定："cpu" / "cuda:0" / "cuda:1"（多GPU场景）
"""
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备: {DEVICE}")


# ============================== 枚举定义 ==============================
class FieldOperation(Enum):
    """字段操作类型枚举"""
    ADD = "add"  # 加法操作（值累加）
    ABS = "abs"  # 绝对值/直接设置操作


# ============================== 核心工具函数 ==============================
def resolve_expr(expr: dict[str, float], names: list[str]) -> list[float]:
    """
    模拟表达式解析函数（支持正则匹配 + 通配符）

    功能说明：
        1. 优先级：.*通配符 > 精确匹配 > 正则匹配
        2. 支持.*_xxx_joint风格的正则模式匹配关节名
        3. 无匹配项时返回0.0

    参数：
        expr: 匹配规则字典，键可为：
              - .*: 通配符（匹配所有名称）
              - 精确名称: 如"joint1"
              - 正则表达式: 如".*_joint"
        names: 需要匹配的目标名称列表（如关节名/执行器名）

    返回：
        按names顺序排列的匹配值列表，无匹配则为0.0
    """
    # 1. 通配符匹配（最高优先级）
    if ".*" in expr:
        return [expr[".*"]] * len(names)

    # 2. 精确匹配 + 正则匹配
    result = []
    for name in names:
        # 优先精确匹配
        if name in expr:
            result.append(expr[name])
            continue

        # 尝试正则匹配
        matched_value = 0.0
        for pattern_key in expr:
            try:
                # 编译正则表达式并完全匹配
                re_pattern = re.compile(pattern_key)
                if re_pattern.fullmatch(name):
                    matched_value = expr[pattern_key]
                    break  # 匹配到立即退出，避免多规则冲突
            except re.error:
                # 非合法正则表达式（普通字符串），跳过
                continue

        result.append(matched_value)

    return result

# ============================== 索引映射类 ==============================
@dataclass(frozen=True)
class MoziEntityIndexing:
    """
    实体索引映射类（不可变数据类）
    功能：存储实体各组件（关节/执行器/传感器/刚体）的索引映射关系
    """
    # 基础ID映射
    body_ids: torch.Tensor  # 刚体ID列表
    ctrl_ids: torch.Tensor  # 控制ID列表（执行器对应）
    joint_ids: torch.Tensor  # 关节ID列表

    # 地址偏移（用于状态张量索引）
    joint_q_adr: torch.Tensor  # 关节位置地址
    joint_v_adr: torch.Tensor  # 关节速度地址
    free_joint_q_adr: torch.Tensor  # 自由关节位置地址
    free_joint_v_adr: torch.Tensor  # 自由关节速度地址

    # 传感器地址映射
    sensor_adr: dict[str, torch.Tensor]  # 传感器名称到地址的映射


# ============================== 实体核心配置类 ==============================
@dataclass
class EntityCfg:
    """
    实体核心配置类
    包含实体的初始状态、规格函数、各类组件配置等
    """

    @dataclass
    class InitialStateCfg:
        """实体初始状态配置子类"""
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 初始位置
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # 初始旋转（四元数）
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 初始线速度
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 初始角速度
        joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})  # 初始关节位置
        joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})  # 初始关节速度

    # 核心配置字段
    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)  # 初始状态配置
    spec_fn: Callable[[], ms_physxMjSpec] = field(
        default_factory=lambda: (lambda: ms_physxMjSpec())
    )  # 仿真规格生成函数
    articulation: EntityArticulationInfoCfg | None = None  # 关节链配置

    # 组件配置列表
    lights: tuple[LightCfg, ...] = field(default_factory=tuple)  # 灯光配置列表
    cameras: tuple[CameraCfg, ...] = field(default_factory=tuple)  # 相机配置列表
    textures: tuple[TextureCfg, ...] = field(default_factory=tuple)  # 纹理配置列表
    materials: tuple[MaterialCfg, ...] = field(default_factory=tuple)  # 材质配置列表
    sensors: tuple[SensorCfg | ContactSensorCfg, ...] = field(default_factory=tuple)  # 传感器配置列表
    collisions: tuple[CollisionCfg, ...] = field(default_factory=tuple)  # 碰撞配置列表

    # 调试配置
    debug_vis: bool = False  # 是否启用调试可视化


@dataclass
class EntityArticulationInfoCfg:
    """
    关节链信息配置类
    包含执行器配置和关节软限制系数
    """
    actuators: tuple[ActuatorCfg, ...] = field(default_factory=tuple)  # 执行器配置列表
    soft_joint_pos_limit_factor: float = 1.0  # 关节位置软限制系数


# ============================== 实体核心类 ==============================
class Entity:
    """
    实体核心类（与Mujoco接口对齐）
    核心功能：
        1. 实体初始化（关节链/传感器/碰撞等）
        2. 状态读写（根状态/关节状态/执行器控制）
        3. 仿真参数配置（刚度/阻尼/力限/关节限制）
        4. 索引映射管理
    """

    def __init__(self, cfg: EntityCfg, robot_prim_paths, device) -> None:
        """
        构造函数
        参数：
            cfg: 实体配置对象
            robot_prim_paths: 机器人Prim路径表达式
            device: 计算设备（cpu/cuda）
        """
        self.cfg = cfg  # 配置对象
        self.device = device  # 计算设备


        # 1. 初始化批量关节链（PhysX核心接口）
        self.batch_artic = BatchArticulation(prim_paths_expr=robot_prim_paths)
        self.batch_artic.set_collision_group()  # 设置碰撞组
        self.batch_artic.apply_contact_sensors()  # 启用接触传感器


        # 3. 初始化实体数据（索引/默认状态/参数等）
        self.initialize(device=self.device)



    # ============================== 属性定义 ==============================
    @property
    def is_fixed_base(self) -> bool:
        """是否为固定基座"""
        return self.batch_artic.is_fixed_base

    @property
    def is_articulated(self) -> bool:
        """是否为关节链实体"""
        return self.batch_artic.is_articulated

    @property
    def is_actuated(self) -> bool:
        """是否包含执行器"""
        return self.batch_artic.is_actuated

    @property
    def is_mocap(self) -> bool:
        """是否为Mocap实体（暂不支持）"""
        return False


    @property
    def data(self) -> EntityData:
        """获取实体数据对象"""
        return self._data

    @property
    def joint_names(self) -> list[str]:
        """获取关节名称列表（映射到执行器名称）"""
        return self.actuator_names

    @property
    def tendon_names(self) -> list[str]:
        """获取肌腱名称列表（暂不支持）"""
        return None

    @property
    def body_names(self) -> list[str]:
        """获取刚体名称列表"""
        return self.batch_artic.rigid_body_names

    @property
    def sensor_names(self) -> list[str]:
        """获取传感器名称列表（暂映射到刚体名称）"""
        return self.body_names

    @property
    def actuator_names(self) -> list[str]:
        """
        获取执行器名称列表
        处理逻辑：从DOF名称中移除后缀"_0"，得到执行器名称
        """
        dof_names = self.batch_artic.dof_names
        actuator_names = []
        for dof_name in dof_names:
            if dof_name.split("_")[-1] == "0":
                actuator_names.append(dof_name.split("_0")[0])
        return actuator_names

    @property
    def num_joints(self) -> int:
        """获取关节数量"""
        return len(self.joint_names)

    @property
    def num_tendons(self) -> int:
        """获取肌腱数量（暂不支持）"""
        return len(self.tendon_names) if self.tendon_names else 0

    @property
    def num_bodies(self) -> int:
        """获取刚体数量"""
        return len(self.body_names)

    @property
    def num_sensors(self) -> int:
        """获取传感器数量"""
        return len(self.sensor_names)

    @property
    def num_actuators(self) -> int:
        """获取执行器数量"""
        return len(self.actuator_names)

    @property
    def root_body(self) -> ms_physxMjsBody:
        """获取根刚体对象"""
        return [0]

    # ============================== 查找方法 ==============================
    def find_bodies(
            self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """
        查找刚体
        参数：
            name_keys: 名称关键字（支持正则/通配符）
            preserve_order: 是否保持输入顺序
        返回：
            (匹配的索引列表, 匹配的名称列表)
        """
        return resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(
            self,
            name_keys: str | Sequence[str],
            joint_subset: list[str] | None = None,
            preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """查找关节（同刚体查找逻辑）"""
        if joint_subset is None:
            joint_subset = self.joint_names
        return resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_tendons(
            self,
            name_keys: str | Sequence[str],
            tendon_subset: list[str] | None = None,
            preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        """查找肌腱（同刚体查找逻辑）"""
        if tendon_subset is None:
            tendon_subset = self.tendon_names
        return resolve_matching_names(name_keys, tendon_subset, preserve_order)

    def find_actuators(
            self,
            name_keys: str | Sequence[str],
            actuator_subset: list[str] | None = None,
            preserve_order: bool = False,
    ):
        """查找执行器（同刚体查找逻辑）"""
        if actuator_subset is None:
            actuator_subset = self.actuator_names
        return resolve_matching_names(name_keys, self.actuator_names, preserve_order)

    def find_sensors(
            self,
            name_keys: str | Sequence[str],
            sensor_subset: list[str] | None = None,
            preserve_order: bool = False,
    ):
        """查找传感器（同刚体查找逻辑）"""
        if sensor_subset is None:
            sensor_subset = self.sensor_names
        return resolve_matching_names(name_keys, sensor_subset, preserve_order)

    # ============================== 文件操作方法 ==============================
    def write_usd(self, xml_path: Path) -> None:
        """写入USD配置文件（暂未实现）"""
        pass

    def to_zip(self, path: Path) -> None:
        """打包为ZIP文件（暂未实现）"""
        pass

    # ============================== 初始化方法 ==============================
    def initialize(
            self,
            device: str,
    ) -> None:
        """
        实体初始化核心方法
        功能：
            1. 计算索引映射
            2. 初始化根状态/关节状态
            3. 配置执行器参数（刚度/阻尼/力限/电枢）
            4. 计算关节软限制
            5. 初始化实体数据对象
        """
        # 1. 计算索引映射
        indexing = self._compute_indexing(device=device)
        self.indexing = indexing
        nworld = self.batch_artic.num_articulations  # 环境数量

        # 2. 初始化根状态
        root_state_components = [self.cfg.init_state.pos, self.cfg.init_state.rot]
        if not self.is_fixed_base:
            root_state_components.extend([self.cfg.init_state.lin_vel, self.cfg.init_state.ang_vel])

        # 转换为张量并扩展到批量维度
        root_state_components = [list(i) for i in root_state_components]
        default_root_state = [torch.tensor(i, dtype=torch.float32, device=self.device)
                              for i in root_state_components]
        default_root_state = torch.cat(default_root_state, axis=0)
        default_root_state = torch.stack([default_root_state for i in range(nworld)])

        # 3. 初始化关节状态
        if self.is_articulated:
            # 初始关节位置/速度
            default_joint_pos = torch.tensor(
                resolve_expr(self.cfg.init_state.joint_pos, self.actuator_names), device=device
            )[None].repeat(nworld, 1)
            default_joint_vel = torch.tensor(
                resolve_expr(self.cfg.init_state.joint_vel, self.actuator_names), device=device
            )[None].repeat(nworld, 1)

            # 执行器参数配置（刚度/阻尼/力限/电枢）
            if self.is_actuated:
                # 初始化参数字典
                robot_stiffness = {}
                robot_damping = {}
                robot_armature = {}
                robot_effort = {}

                # 从执行器配置中提取参数
                for a in self.cfg.articulation.actuators:
                    names = a.joint_names_expr
                    for name in names:
                        robot_effort[name] = a.effort_limit
                        robot_stiffness[name] = a.stiffness
                        robot_damping[name] = a.damping
                        robot_armature[name] = a.armature

                # 解析参数到关节维度
                stiffness_mozi = resolve_expr(robot_stiffness, self.joint_names)
                damping_mozi = resolve_expr(robot_damping, self.joint_names)
                effort_limit_mozi = resolve_expr(robot_effort, self.joint_names)
                armature_mozi = resolve_expr(robot_armature, self.joint_names)

                # 设置到PhysX引擎
                self.batch_artic.set_dof_drive_params_stiffness(
                    np.array([stiffness_mozi for i in range(nworld)])
                )
                self.batch_artic.set_dof_drive_params_dampings(
                    np.array([damping_mozi for i in range(nworld)])
                )
                self.batch_artic.set_dof_drive_params_max_effort(
                    np.array([effort_limit_mozi for i in range(nworld)]) * 1.5
                )
                self.batch_artic.set_dof_armatures(
                    np.array([armature_mozi for i in range(nworld)])
                )

                # 获取驱动参数并提取刚度/阻尼
                dof_params = torch.tensor(
                    self.batch_artic.get_dof_drive_params(),
                    dtype=torch.float32,
                    device=self.device
                )
                default_joint_stiffness = dof_params[:, self.indexing.ctrl_ids, 0]
                default_joint_damping = dof_params[:, self.indexing.ctrl_ids, 1]
            else:
                # 无执行器时初始化空张量
                default_joint_stiffness = torch.empty(
                    nworld, 0, dtype=torch.float, device=device
                )
                default_joint_damping = torch.empty(
                    nworld, 0, dtype=torch.float, device=device
                )

            # 4. 关节限制处理
            dof_limits = torch.tensor(self.batch_artic.dof_limits, dtype=torch.float32, device=device)
            default_joint_pos_limits = dof_limits.clone()
            joint_pos_limits = default_joint_pos_limits.clone()

            # 计算关节位置均值和范围
            joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
            joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

            # 计算软限制
            soft_limit_factor = (
                self.cfg.articulation.soft_joint_pos_limit_factor
                if self.cfg.articulation
                else 1.0
            )
            soft_joint_pos_limits = torch.stack(
                [
                    joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
                    joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
                ],
                dim=-1,
            )
        else:
            # 非关节链实体：初始化空张量
            empty_shape = (nworld, 0)
            default_joint_pos = torch.empty(*empty_shape, dtype=torch.float, device=device)
            default_joint_vel = torch.empty(*empty_shape, dtype=torch.float, device=device)
            default_joint_stiffness = torch.empty(*empty_shape, dtype=torch.float, device=device)
            default_joint_damping = torch.empty(*empty_shape, dtype=torch.float, device=device)
            default_joint_pos_limits = torch.empty(*empty_shape, 2, dtype=torch.float, device=device)
            joint_pos_limits = torch.empty(*empty_shape, 2, dtype=torch.float, device=device)
            soft_joint_pos_limits = torch.empty(*empty_shape, 2, dtype=torch.float, device=device)

        # 5. 初始化实体数据对象
        self._data = EntityData(
            indexing=indexing,
            batch_artic=self.batch_artic,
            device=self.device,
            default_root_state=default_root_state,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            default_joint_stiffness=default_joint_stiffness,
            default_joint_damping=default_joint_damping,
            default_joint_pos_limits=default_joint_pos_limits,
            joint_pos_limits=joint_pos_limits,
            soft_joint_pos_limits=soft_joint_pos_limits,
            gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(nworld, 1),
            forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(nworld, 1),
            is_fixed_base=self.is_fixed_base,
            is_articulated=self.is_articulated,
            is_actuated=self.is_actuated,
        )

        # 6. 初始化字段读写函数映射
        self.field_set_funcs = {
            "body_mass": self.data.set_rigid_body_mass,
            "body_friction": self.data.set_body_link_friction_params,
        }
        self.field_get_funcs = {
            "body_mass": self.data.get_rigid_body_mass,
            "body_friction": self.data.get_body_link_friction_params,
        }

    # ============================== 参数获取方法 ==============================
    def get_stiffness(self):
        """获取关节刚度张量"""
        return torch.tensor(
            self.batch_artic.get_dof_drive_params()[:, :, 0],
            dtype=torch.float32,
            device=self.device
        )

    def get_damping(self):
        """获取关节阻尼张量（取负值）"""
        return torch.tensor(
            -self.batch_artic.get_dof_drive_params()[:, :, 1],
            dtype=torch.float32,
            device=self.device
        )

    # ============================== 状态更新方法 ==============================
    def update(self, dt: float) -> None:
        """状态更新（暂未实现）"""
        del dt  # 未使用参数

    def reset(self, env_ids: Optional[Union[torch.Tensor, slice]] = None) -> None:
        """重置实体状态"""
        self.clear_state(env_ids)

    def write_data_to_sim(self) -> None:
        """将数据写入仿真（暂未实现）"""
        pass

    def clear_state(self, env_ids: Optional[Union[torch.Tensor, slice]] = None) -> None:
        """清空实体状态"""
        self._data.clear_state(env_ids)

    # ============================== 状态写入方法 ==============================
    def write_root_state_to_sim(
            self, root_state: torch.Tensor, env_ids: Optional[Union[torch.Tensor, slice]] = None
    ) -> None:
        """写入根状态到仿真"""
        self._data.write_root_state(root_state, env_ids)

    def write_root_link_pose_to_sim(
            self,
            root_pose: torch.Tensor,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入根连杆位姿到仿真"""
        self._data.write_root_pose(root_pose, env_ids)

    def write_root_link_velocity_to_sim(
            self,
            root_velocity: torch.Tensor,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入根连杆速度到仿真"""
        self._data.write_root_velocity(root_velocity, env_ids)

    def write_joint_state_to_sim(
            self,
            position: torch.Tensor,
            velocity: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入关节状态（位置+速度）到仿真"""
        self._data.write_joint_state(position, velocity, joint_ids, env_ids)

    def write_joint_efforts_to_sim(
            self,
            efforts: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入关节力到仿真"""
        self._data.write_joint_efforts(efforts, joint_ids, env_ids)

    def write_joint_position_to_sim(
            self,
            position: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入关节位置到仿真"""
        self._data.write_joint_position(position, joint_ids, env_ids)

    def write_joint_velocity_to_sim(
            self,
            velocity: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ):
        """写入关节速度到仿真"""
        self._data.write_joint_velocity(velocity, joint_ids, env_ids)

    def write_joint_position_target_to_sim(
            self,
            position_target: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ) -> None:
        """写入关节位置目标到仿真（控制量）"""
        self._data.write_ctrl(position_target, joint_ids, env_ids)

    def write_joint_velocity_target_to_sim(
            self,
            velocity_target: torch.Tensor,
            joint_ids: Optional[Union[torch.Tensor, slice]] = None,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ) -> None:
        """写入关节速度目标到仿真"""
        self._data.write_velocity(velocity_target, joint_ids, env_ids)

    def write_external_wrench_to_sim(
            self,
            forces: torch.Tensor,
            torques: torch.Tensor,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
            body_ids: Optional[Union[Sequence[int], slice]] = None,
    ) -> None:
        """写入外部力/力矩到仿真"""
        self._data.write_external_wrench(forces, torques, body_ids, env_ids)

    def set_field(self, env_ids, values, field, operation, idx_list):
        """
        设置实体字段值（支持加法/直接设置操作）
        参数：
            env_ids: 环境ID列表
            values: 要设置的值张量
            field: 字段名称（如body_mass/body_friction）
            operation: 操作类型（add/abs）
            idx_list: 目标索引列表
        """
        # 校验字段是否支持
        if field not in self.field_set_funcs or field not in self.field_get_funcs:
            raise ValueError(
                f"字段 {field} 不支持，支持的字段列表：{list(self.field_set_funcs.keys())}"
            )

        # 转换索引到张量
        idx_list = torch.tensor(
            idx_list,
            dtype=torch.int,
            device=self.device
        ).squeeze()

        # 处理值张量维度
        value_dim = values.shape[0]
        if value_dim == 1:
            # 单维度值：挤压所有非第0维
            squeeze_dims = list(range(1, values.dim()))
            values = values.squeeze(dim=squeeze_dims)
        else:
            # 多维度值：挤压后按环境ID索引
            values = values.squeeze()[env_ids]

        # 执行字段设置操作
        if operation == "add":
            # 加法操作：当前值 + 新值
            self.field_set_funcs[field](
                values + self.field_get_funcs[field]()[env_ids, idx_list],
                env_ids,
                idx_list,
            )
        elif operation == "abs":
            # 直接设置操作
            self.field_set_funcs[field](
                values,
                env_ids,
                idx_list,
            )
        else:
            raise NotImplementedError(f"不支持的操作类型：{operation}")

    def write_mocap_pose_to_sim(
            self,
            mocap_pose: torch.Tensor,
            env_ids: Optional[Union[torch.Tensor, slice]] = None,
    ) -> None:
        """写入Mocap位姿到仿真（暂未实现）"""
        self._data.write_mocap_pose(mocap_pose, env_ids)

    # ============================== 私有方法 ==============================
    def _compute_indexing(self, device: str) -> EntityIndexing:
        """
        计算实体索引映射
        功能：构建关节/执行器/刚体/传感器的索引关系
        """
        # 初始化基础ID张量
        body_ids = torch.zeros(self.num_bodies, dtype=torch.int, device=device)
        joint_ids = torch.zeros(self.num_joints, dtype=torch.int, device=device)

        # 初始化控制ID（执行器ID）
        if self.is_actuated:
            ctrl_names = []
            for a in self.cfg.articulation.actuators:
                names = a.joint_names_expr
                for name in names:
                    ctrl_names.append(name)
            ctrl_ids = torch.tensor([i for i in range(self.num_actuators)], dtype=torch.int, device=device)
        else:
            ctrl_ids = torch.empty(0, dtype=torch.int, device=device)

        # 初始化地址偏移张量（空张量）
        joint_q_adr = torch.tensor([], dtype=torch.int, device=device)
        joint_v_adr = torch.tensor([], dtype=torch.int, device=device)
        free_joint_q_adr = torch.tensor([], dtype=torch.int, device=device)
        free_joint_v_adr = torch.tensor([], dtype=torch.int, device=device)

        # 初始化传感器地址映射
        sensor_adr = {}

        # 返回索引映射对象
        return MoziEntityIndexing(
            body_ids=body_ids,
            ctrl_ids=ctrl_ids,
            joint_ids=joint_ids,
            joint_q_adr=joint_q_adr,
            joint_v_adr=joint_v_adr,
            free_joint_q_adr=free_joint_q_adr,
            free_joint_v_adr=free_joint_v_adr,
            sensor_adr=sensor_adr,
        )


# ============================== 使用示例 ==============================
if __name__ == "__main__":
    # 创建实体配置
    cfg = EntityCfg()
    cfg.articulation = EntityArticulationInfoCfg(
        actuators=(ActuatorCfg(name="act1"), ActuatorCfg(name="act2"))
    )

    # 创建实体实例（注意：实际使用需要传入robot_prim_paths和device参数）
    # entity = Entity(cfg, robot_prim_paths="/World/Robot", device=DEVICE)

    # 测试属性访问（示例）
    # print(f"关节数量: {entity.num_joints}")
    # print(f"执行器数量: {entity.num_actuators}")
    # print(f"根状态维度: {entity.data.ROOT_STATE_DIM}")

    # 测试状态写入（示例）
    # num_envs = 4
    # new_root_state = torch.randn(num_envs, entity.data.ROOT_STATE_DIM, device=DEVICE)
    # entity.write_root_state_to_sim(new_root_state)

    # new_joint_pos = torch.randn(num_envs, entity.num_joints, device=DEVICE)
    # entity.write_joint_position_to_sim(new_joint_pos)

    print("代码规范化完成，示例代码需补充实际参数后运行")

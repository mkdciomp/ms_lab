"""
Mujoco后端模块 - 用于场景管理、仿真控制和域随机化
核心功能：
1. 初始化和管理Mujoco仿真场景与仿真器
2. 提供模型参数的域随机化能力（关节、刚体、几何等属性）
3. 封装仿真步长、重置、更新等核心操作
"""
from ms_lab.scene import Scene
from ms_lab.sim.sim import Simulation
from ms_lab.managers.scene_entity_config import SceneEntityCfg
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
from ms_lab.entity import Entity, EntityIndexing
import torch
from dataclasses import dataclass
from ms_lab.third_party.isaaclab.isaaclab.utils.math import (
    quat_apply_inverse,
    sample_log_uniform,
    sample_uniform,
)

# 默认机器人资产配置
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class FieldSpec:
    """
    字段规格说明类 - 定义如何处理特定的模型字段

    属性:
        entity_type: 实体类型 (自由度/关节/刚体/几何/站点/执行器)
        use_address: 是否使用地址索引 (True用于需要q_adr/v_adr的字段)
        default_axes: 默认要随机化的轴索引列表
        valid_axes: 该字段允许随机化的有效轴索引列表
    """
    entity_type: Literal["dof", "joint", "body", "geom", "site", "actuator"]
    use_address: bool = False  # True表示需要地址索引（q_adr, v_adr）
    default_axes: Optional[List[int]] = None
    valid_axes: Optional[List[int]] = None


# 预定义的字段规格映射 - 覆盖所有支持随机化的模型字段
FIELD_SPECS = {
    # 自由度相关 - 使用地址索引
    "dof_armature": FieldSpec("dof", use_address=True),  # 自由度电枢
    "dof_frictionloss": FieldSpec("dof", use_address=True),  # 自由度摩擦损耗
    "dof_damping": FieldSpec("dof", use_address=True),  # 自由度阻尼
    # 关节相关 - 直接使用ID
    "jnt_range": FieldSpec("joint"),  # 关节范围
    "jnt_stiffness": FieldSpec("joint"),  # 关节刚度
    # 刚体相关 - 直接使用ID
    "body_mass": FieldSpec("body"),  # 刚体质量
    "body_ipos": FieldSpec("body", default_axes=[0, 1, 2]),  # 刚体初始位置
    "body_iquat": FieldSpec("body", default_axes=[0, 1, 2, 3]),  # 刚体初始四元数
    "body_inertia": FieldSpec("body"),  # 刚体惯性
    "body_pos": FieldSpec("body", default_axes=[0, 1, 2]),  # 刚体位置
    "body_quat": FieldSpec("body", default_axes=[0, 1, 2, 3]),  # 刚体四元数
    # 几何相关 - 直接使用ID
    "geom_friction": FieldSpec("geom", default_axes=[0], valid_axes=[0, 1, 2]),  # 几何摩擦
    "geom_pos": FieldSpec("geom", default_axes=[0, 1, 2]),  # 几何位置
    "geom_quat": FieldSpec("geom", default_axes=[0, 1, 2, 3]),  # 几何四元数
    "geom_rgba": FieldSpec("geom", default_axes=[0, 1, 2, 3]),  # 几何颜色(RGBA)
    # 站点相关 - 直接使用ID
    "site_pos": FieldSpec("site", default_axes=[0, 1, 2]),  # 站点位置
    "site_quat": FieldSpec("site", default_axes=[0, 1, 2, 3]),  # 站点四元数
    # 特殊字段 - 使用地址索引
    "qpos0": FieldSpec("joint", use_address=True),  # 关节初始位置
}


def _get_entity_indices(
        indexing: EntityIndexing, asset_cfg: SceneEntityCfg, spec: FieldSpec
) -> torch.Tensor:
    """
    获取实体索引张量 - 根据实体类型和配置返回对应的索引

    参数:
        indexing: 实体索引对象
        asset_cfg: 资产配置
        spec: 字段规格

    返回:
        实体索引张量
    """
    match spec.entity_type:
        case "dof":
            return indexing.joint_v_adr[asset_cfg.joint_ids]
        case "joint" if spec.use_address:
            return indexing.joint_q_adr[asset_cfg.joint_ids]
        case "joint":
            return indexing.joint_ids[asset_cfg.joint_ids]
        case "body":
            return indexing.body_ids[asset_cfg.body_ids]
        case "geom":
            return indexing.geom_ids[asset_cfg.geom_ids]
        case "site":
            return indexing.site_ids[asset_cfg.site_ids]
        case "actuator":
            assert indexing.ctrl_ids is not None, "执行器控制ID不能为空"
            return indexing.ctrl_ids[asset_cfg.actuator_ids]
        case _:
            raise ValueError(f"未知的实体类型: {spec.entity_type}")


def _determine_target_axes(
        model_field: torch.Tensor,
        spec: FieldSpec,
        axes: Optional[List[int]],
        ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
) -> List[int]:
    """
    确定要随机化的目标轴索引

    参数:
        model_field: 模型字段张量
        spec: 字段规格
        axes: 用户指定的轴索引列表
        ranges: 随机化范围（元组或字典）

    返回:
        目标轴索引列表
    """
    # 计算字段维度（减去环境维度）
    field_ndim = len(model_field.shape) - 1

    if axes is not None:
        # 使用用户显式指定的轴
        target_axes = axes
    elif isinstance(ranges, dict):
        # 从范围字典的键获取轴索引
        target_axes = list(ranges.keys())
    elif spec.default_axes is not None:
        # 使用字段规格的默认轴
        target_axes = spec.default_axes
    else:
        # 随机化所有轴
        if field_ndim > 1:
            target_axes = list(range(model_field.shape[-1]))  # 最后一维的所有轴
        else:
            target_axes = [0]  # 标量字段

    # 验证轴的有效性
    if spec.valid_axes is not None:
        invalid_axes = set(target_axes) - set(spec.valid_axes)
        if invalid_axes:
            raise ValueError(
                f"字段的无效轴索引 {invalid_axes}，有效轴索引: {spec.valid_axes}"
            )

    return target_axes


def _prepare_axis_ranges(
        ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]],
        target_axes: List[int],
        field: str,
) -> Dict[int, Tuple[float, float]]:
    """
    将随机化范围转换为统一的字典格式

    参数:
        ranges: 随机化范围（元组表示所有轴相同范围，字典表示各轴独立范围）
        target_axes: 目标轴索引列表
        field: 字段名称

    返回:
        轴索引到范围的映射字典
    """
    if isinstance(ranges, tuple):
        # 所有轴使用相同的范围
        return {axis: ranges for axis in target_axes}
    elif isinstance(ranges, dict):
        # 验证所有目标轴都有对应的范围定义
        missing_axes = set(target_axes) - set(ranges.keys())
        if missing_axes:
            raise ValueError(
                f"字段 '{field}' 缺少轴 {missing_axes} 的范围定义，"
                f"需要定义的轴: {target_axes}"
            )
        return {axis: ranges[axis] for axis in target_axes}
    else:
        raise TypeError(f"范围必须是元组或字典类型，当前类型: {type(ranges)}")


def _generate_random_values(
        distribution: str,
        axis_ranges: Dict[int, Tuple[float, float]],
        indexed_data: torch.Tensor,
        target_axes: List[int],
        device: str,
) -> torch.Tensor:
    """
    为指定轴生成随机值

    参数:
        distribution: 分布类型 (uniform/log_uniform/gaussian)
        axis_ranges: 轴范围字典
        indexed_data: 索引数据张量
        target_axes: 目标轴列表
        device: 计算设备

    返回:
        随机值张量
    """
    result = indexed_data.clone()

    for axis in target_axes:
        lower, upper = axis_ranges[axis]

        # 转换为张量并移动到指定设备
        lower_bound = torch.tensor(lower, device=device)
        upper_bound = torch.tensor(upper, device=device)

        # 确定随机值形状
        if len(indexed_data.shape) > 2:  # 多维字段
            shape = (*indexed_data.shape[:-1], 1)  # 保持前n-1维，最后一维为1
        else:  # 标量或二维字段
            shape = indexed_data.shape

        # 采样随机值
        random_vals = _sample_distribution(
            distribution, lower_bound, upper_bound, shape, device
        )

        # 将随机值赋值到对应轴
        if len(indexed_data.shape) > 2:
            result[..., axis] = random_vals.squeeze(-1)
        else:
            result = random_vals

    return result


def _apply_operation(
        model_field: torch.Tensor,
        env_grid: torch.Tensor,
        entity_grid: torch.Tensor,
        indexed_data: torch.Tensor,
        random_values: torch.Tensor,
        operation: str,
):
    """
    应用随机化操作到模型字段

    参数:
        model_field: 模型字段张量
        env_grid: 环境网格索引
        entity_grid: 实体网格索引
        indexed_data: 原始数据
        random_values: 随机值
        operation: 操作类型 (add/scale/abs)
    """
    if operation == "add":
        # 加法操作：原始值 + 随机值
        model_field[env_grid, entity_grid] = indexed_data + random_values
    elif operation == "scale":
        # 缩放操作：原始值 * 随机值
        model_field[env_grid, entity_grid] = indexed_data * random_values
    elif operation == "abs":
        # 替换操作：直接使用随机值
        model_field[env_grid, entity_grid] = random_values
    else:
        raise ValueError(f"未知的操作类型: {operation}")


def _sample_distribution(
        distribution: str,
        lower: torch.Tensor,
        upper: torch.Tensor,
        shape: tuple,
        device: str,
) -> torch.Tensor:
    """
    从指定分布中采样

    参数:
        distribution: 分布类型
        lower: 下界
        upper: 上界
        shape: 输出形状
        device: 计算设备

    返回:
        采样结果张量
    """
    if distribution == "uniform":
        return sample_uniform(lower, upper, shape, device=device)
    elif distribution == "log_uniform":
        return sample_log_uniform(lower, upper, shape, device=device)
    elif distribution == "gaussian":
        # 注意：此处原代码缺少sample_gaussian函数定义，需确认实现
        raise NotImplementedError("高斯分布采样尚未实现")
    else:
        raise ValueError(f"未知的分布类型: {distribution}")


class MujocoBackend:
    """
    Mujoco仿真后端类 - 封装场景和仿真器的核心操作

    主要功能:
    1. 场景和仿真器初始化
    2. 模型参数域随机化
    3. 仿真步长控制
    4. 场景重置和更新
    """

    def __init__(self, device: str):
        """
        初始化Mujoco后端

        参数:
            device: 计算设备 (cpu/cuda)
        """
        self.device = device
        self._scene: Optional[Scene] = None
        self._sim: Optional[Simulation] = None

    @property
    def scene(self) -> Scene:
        """获取场景对象"""
        return self._scene

    @property
    def sim(self) -> Simulation:
        """获取仿真器对象"""
        return self._sim

    @property
    def num_envs(self) -> int:
        """获取环境数量"""
        return self.scene.num_envs

    def _init_scene(self, cfg):
        """
        初始化场景

        参数:
            cfg: 场景配置
        """
        self._scene = Scene(cfg, device=self.device)

    def _init_sim(self, cfg):
        """
        初始化仿真器

        参数:
            cfg: 仿真配置
        """
        # 编辑场景规格
        cfg.mujoco.edit_spec(self._scene.spec)

        # 创建仿真器实例
        self._sim = Simulation(
            num_envs=self._scene.num_envs,
            cfg=cfg,
            model=self._scene.compile(),
            device=self.device,
        )

        # 初始化场景
        self._scene.initialize(
            mj_model=self._sim.mj_model,
            model=self._sim.model,
            data=self._sim.data,
        )

    def expand_model_fields(self, domain_randomization_fields):
        """
        扩展模型字段以支持域随机化

        参数:
            domain_randomization_fields: 域随机化字段列表
        """
        self._sim.expand_model_fields(domain_randomization_fields)

    def create_graph(self):
        """创建仿真计算图"""
        self._sim.create_graph()

    def get_robot(self, asset_name: str) -> Entity:
        """
        获取指定名称的机器人实体

        参数:
            asset_name: 资产名称

        返回:
            机器人实体对象
        """
        return self._scene[asset_name]

    def random_field(
            self,
            env_ids: Optional[torch.Tensor] = None,
            field: str = "",
            ranges: Union[Tuple[float, float], Dict[int, Tuple[float, float]]] = (0.0, 1.0),
            distribution: str = "uniform",
            operation: str = "abs",
            asset_cfg: Optional[SceneEntityCfg] = None,
            axes: Optional[List[int]] = None,
    ):
        """
        对模型字段进行随机化

        参数:
            env_ids: 要随机化的环境ID列表 (None表示所有环境)
            field: 要随机化的字段名称
            ranges: 随机化范围
            distribution: 分布类型
            operation: 操作类型 (add/scale/abs)
            asset_cfg: 资产配置
            axes: 要随机化的轴索引列表
        """
        # 获取字段规格
        spec = FIELD_SPECS[field]
        # 使用默认配置或指定配置
        asset_cfg = asset_cfg or _DEFAULT_ASSET_CFG
        # 获取机器人资产
        asset = self.get_robot(asset_cfg.name)

        # 处理环境ID
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.int)

        # 获取模型字段
        model_field = self.get_model_field(field)

        # 获取实体索引
        entity_indices = _get_entity_indices(asset.indexing, asset_cfg, spec)

        # 确定目标轴
        target_axes = _determine_target_axes(model_field, spec, axes, ranges)

        # 准备轴范围
        axis_ranges = _prepare_axis_ranges(ranges, target_axes, field)

        # 创建环境-实体网格索引
        env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")

        # 获取索引数据 (注意：entity_grid需要转到CPU以匹配模型字段的索引)
        indexed_data = model_field[env_grid, entity_grid.cpu()]

        # 生成随机值
        random_values = _generate_random_values(
            distribution, axis_ranges, indexed_data, target_axes, self.device
        )

        # 应用随机化操作
        _apply_operation(
            model_field, env_grid, entity_grid, indexed_data, random_values, operation
        )

    def get_all_robots(self):
        """
        获取所有机器人实体

        返回:
            所有机器人实体的迭代器
        """
        # 修复拼写错误: _cene -> _scene
        return self._scene.entities.values()

    def get_terrain(self):
        """获取地形对象"""
        return self._scene.terrain

    def get_env_origins(self):
        """获取环境原点坐标"""
        return self._scene.env_origins

    def write_data_to_sim(self):
        """将场景数据写入仿真器"""
        self._scene.write_data_to_sim()

    def forward(self):
        """执行仿真前向计算"""
        self._sim.forward()

    def step(self):
        """执行仿真步长"""
        self._sim.step()

    def update(self, dt: float):
        """
        更新场景

        参数:
            dt: 时间步长
        """
        self._scene.update(dt)

    def reset(self, env_ids: torch.Tensor):
        """
        重置指定环境

        参数:
            env_ids: 要重置的环境ID列表
        """
        self._scene.reset(env_ids)

    def get_model_field(self, field: str) -> torch.Tensor:
        """
        获取模型字段

        参数:
            field: 字段名称

        返回:
            字段张量
        """
        return getattr(self._sim.model, field)

    def get_stiffness(self, ctrl_ids: torch.Tensor) -> torch.Tensor:
        """
        获取执行器刚度

        参数:
            ctrl_ids: 控制ID列表

        返回:
            刚度值张量
        """
        return self._sim.mj_model.actuator_gainprm[ctrl_ids, 0]

    def get_damping(self, ctrl_ids: torch.Tensor) -> torch.Tensor:
        """
        获取执行器阻尼 (修复原代码拼写错误: dampling -> damping)

        参数:
            ctrl_ids: 控制ID列表

        返回:
            阻尼值张量
        """
        return self._sim.mj_model.actuator_biasprm[ctrl_ids, 2]

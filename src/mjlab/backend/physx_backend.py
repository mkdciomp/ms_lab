import numpy as np
import torch
# 导入ms_lab的PhysX仿真模块
from ms_lab.ms_physx.sim import Simulation
# 导入Mozisim仿真应用核心模块
from mozisim.core.simulation_app import SimulationApp
# 导入场景管理模块
from ms_lab.ms_physx.scene import Scene
# 导入实体（机器人/物体）管理模块
from ms_lab.ms_physx.entity import Entity
# 导入Mozisim工具类和场景保存工具
from mozisim.utils.mozi_utils import MoziUtils
from mozisim.utils.stage_utils import save_stage
# 导入IsaacLab的数学采样工具
from ms_lab.third_party.isaaclab.isaaclab.utils.math import (
    sample_log_uniform,  # 对数均匀采样
    sample_uniform,  # 均匀采样
)

# 定义域随机化字段的维度配置
# body_mass: 质量，标量（维度1）
# foot_friction: 脚部摩擦系数，向量（维度3）
FIELD_DIM = {
    "body_mass": 1,
    "foot_friction": 3,
}


def _sample_distribution(
        distribution: str,
        lower: torch.Tensor,
        upper: torch.Tensor,
        shape: tuple,
        device: str,
) -> torch.Tensor:
    """
    从指定的分布中采样数据
    参数:
        distribution: 分布类型，支持 "uniform"(均匀)、"log_uniform"(对数均匀)、"gaussian"(高斯)
        lower: 采样下界（张量）
        upper: 采样上界（张量）
        shape: 输出数据的形状
        device: 计算设备（cpu/cuda）
    返回:
        采样得到的张量
    """
    if distribution == "uniform":
        return sample_uniform(lower, upper, shape, device=device)
    elif distribution == "log_uniform":
        return sample_log_uniform(lower, upper, shape, device=device)
    elif distribution == "gaussian":
        # 注意：原代码中未导入sample_gaussian，实际使用时需要补充导入
        return sample_gaussian(lower, upper, shape, device=device)
    else:
        raise ValueError(f"未知的分布类型: {distribution}")


class PhysxBackend:
    """
    PhysX物理仿真后端类
    负责管理仿真场景、实体、物理计算和渲染等核心功能
    """

    def __init__(self, render_mode, device):
        """
        初始化PhysX仿真后端
        参数:
            render_mode: 渲染模式
                - "human": 人类可视化模式（有窗口，非无头）
                - "rgb_array": 图像数组模式（无头，输出图像数据）
                - 其他: 纯物理仿真（无头，无渲染）
            device: 计算设备（cpu/cuda）
        """
        self.device = device  # 保存计算设备
        self.utils = MoziUtils()  # 初始化Mozisim工具类

        self.entities = {}  # 存储仿真中的实体（机器人）字典 {实体名: Entity对象}
        # 根据渲染模式配置渲染参数
        if render_mode == "human":
            self.is_headless = False  # 非无头模式（显示窗口）
            self.is_render = True  # 启用渲染
        elif render_mode == "rgb_array":
            self.is_headless = True  # 无头模式（无窗口）
            self.is_render = True  # 启用渲染（输出图像）
        else:
            self.is_headless = True  # 无头模式
            self.is_render = False  # 禁用渲染

        # 初始化仿真应用
        self.app = SimulationApp(config={"headless": self.is_headless, "is_output_texture_buff": self.is_render})

        self.terrain = None  # 地形对象，后续初始化
        self.objects = {}  # 存储场景中的物体字典 {物体名: 物体对象}

    # 以下为属性访问器，提供对私有变量的安全访问
    @property
    def scene(self):
        """获取场景对象"""
        return self._scene

    @property
    def sim(self):
        """获取仿真对象"""
        return self._sim

    @property
    def num_envs(self):
        """获取环境数量（并行仿真的环境数）"""
        return self.scene.num_envs

    def _init_scene(self, cfg):
        """
        初始化仿真场景
        参数:
            cfg: 配置对象，包含场景、实体、地形等配置信息
        """
        # 设置地形的环境数量与总环境数一致
        cfg.terrain.num_envs = cfg.num_envs

        # 创建场景对象
        self._scene = Scene(cfg, device=self.device)
        # 保存实体和物体配置
        self.ent_cfgs = cfg.entities  # 实体（机器人）配置
        self.object_cfgs = cfg.objects  # 物体配置
        # 添加地形到场景
        self.scene.add_terrain()
        # 存储实体的USD路径和配置
        self.ent_prim_dict = {}  # {实体名: 实体USD路径列表}
        self.ent_cfg_dict = {}  # {实体名: 实体配置}

        # 遍历所有实体配置，将机器人添加到场景
        for ent_name, ent_cfg in self.ent_cfgs.items():
            # 添加机器人到场景，返回其USD路径
            ent_prim_paths = self.scene.add_robot(ent_name, ent_cfg)
            self.ent_prim_dict[ent_name] = ent_prim_paths
            self.ent_cfg_dict[ent_name] = ent_cfg

        # 保存地形对象
        self.terrain = self.scene.terrain
        # 初始化默认的环境原点（每个环境的初始位置）
        self._default_env_origins = torch.zeros(
            (cfg.num_envs, 3), device=self.device, dtype=torch.float32
        )

    def _init_sim(self, cfg):
        """
        初始化物理仿真器
        参数:
            cfg: 仿真配置对象
        """
        # 创建仿真对象
        self._sim = Simulation(
            cfg=cfg,
            device=self.device,
            is_render=self.is_render,
        )
        # 为每个实体创建Entity对象并添加到仿真
        for ent_name, ent_prim_paths in self.ent_prim_dict.items():
            ent_cfg = self.ent_cfg_dict[ent_name]
            # 创建实体对象
            ent = Entity(ent_cfg, ent_prim_paths, device=self.device)
            self.entities[ent_name] = ent

        # 如果有物体配置，添加物体到场景
        if self.object_cfgs is not None:
            for object_name, object_cfg in self.object_cfgs.items():
                object_cfg.num_envs = self.num_envs  # 设置物体的环境数量
                # 添加物体到场景
                object = self.scene.add_object(object_name, object_cfg)
                self.objects[object_name] = object

        # 重新定义域随机化字段维度（覆盖全局FIELD_DIM，可能是笔误）
        # 注意：此处body_friction应为foot_friction，与全局配置保持一致
        self.field_dim = {
            "body_mass": 1,
            "body_friction": 3,
        }
        # 保存场景到USD文件（调试用，已注释）
        # save_stage("./rough_scene_0107_3072.usda")
        # exit()

    def expand_model_fields(self, domain_randomization_fields):
        """
        扩展仿真模型的域随机化字段
        参数:
            domain_randomization_fields: 需要随机化的字段列表
        """
        self._sim.expand_model_fields(domain_randomization_fields)

    def create_graph(self):
        """创建仿真计算图（PhysX的GPU计算图）"""
        self._sim.create_graph()

    def get_robot(self, asset_name):
        """
        获取指定名称的机器人实体
        参数:
            asset_name: 机器人名称
        返回:
            对应的Entity对象
        """
        return self.entities[asset_name]

    def get_all_robots(self):
        """获取所有机器人实体"""
        return self.entities.values()

    def get_terrain(self):
        """获取地形对象"""
        return self.terrain

    def get_env_origins(self):
        """
        获取所有环境的原点坐标
        返回:
            形状为[num_envs, 3]的张量，每个环境的x/y/z原点坐标
        """
        # 如果有地形，返回地形定义的环境原点
        if self.terrain is not None:
            assert self.terrain.env_origins is not None
            return self.terrain.env_origins
        # 否则返回默认的原点（全零）
        assert self._default_env_origins is not None
        return self._default_env_origins
        # 以下代码不可达，建议删除
        return self._scene.env_origins

    def write_data_to_sim(self):
        """将数据写入仿真器（预留接口，暂未实现）"""
        pass
        # self._scene.write_data_to_sim()

    def forward(self):
        """执行仿真前向计算（更新物理状态）"""
        self._sim.forward()

    def step(self):
        """执行仿真步（物理引擎步进）"""
        self._sim.step()

    def update(self, dt):
        """
        更新仿真状态
        参数:
            dt: 时间步长（秒）
        """
        self._sim.update(dt)

    def reset(self, env_ids):
        """
        重置指定环境的状态
        参数:
            env_ids: 需要重置的环境ID列表/张量
        """
        self._scene.reset(env_ids)

    def get_model_field(self, field):
        """
        获取仿真模型的指定字段值
        参数:
            field: 字段名（如body_mass, foot_friction）
        返回:
            模型对应字段的值
        """
        # pass
        return getattr(self._sim.model, field)

    def ramdom_values(self, ranges, distribution):
        """
        生成随机值（方法名拼写错误：ramdom -> random，建议修正）
        参数:
            ranges: 随机值范围 [min, max]
            distribution: 分布类型
        返回:
            固定形状[2,2]的随机张量（示例实现，实际应根据需求修改）
        """
        return torch.randn([2, 2], dtype=torch.float32, device=self.device)

    def random_field(
            self,
            env_ids,
            field,
            ranges,
            distribution,
            operation,
            asset_cfg,
            axes
    ):
        """
        对指定字段进行域随机化（Domain Randomization）
        参数:
            env_ids: 需要随机化的环境ID，None表示所有环境
            field: 要随机化的字段名（如body_mass, foot_friction）
            ranges: 随机值范围 [lower, upper]
            distribution: 采样分布类型
            operation: 操作类型（如赋值、缩放等）
            asset_cfg: 资产配置（包含机器人名称、身体部位名称等）
            axes: 轴配置（未使用）
        """
        # 如果未指定环境ID，使用所有环境
        if env_ids is None:
            env_ids = torch.tensor(
                [i for i in range(self.num_envs)],
                device=self.device, dtype=torch.int)

        # 获取目标机器人实体
        ent = self.get_robot(asset_cfg.name)

        # 找到需要随机化的身体部位索引
        # 匹配实体的身体部位名称和配置中的名称
        idx_list = (idx for idx, val in enumerate(ent.body_names) if val in asset_cfg.body_names)
        idx_list = list(idx_list)

        # 定义输出形状: [环境数, 身体部位数, 字段维度]
        shape = [self.num_envs, len(idx_list), self.field_dim[field]]

        # 从指定分布采样随机值
        values = _sample_distribution(
            distribution,
            ranges[0],  # 下界
            ranges[1],  # 上界
            torch.tensor(shape, device=self.device),  # 形状
            device=self.device  # 设备
        )

        # 将随机值应用到实体的指定字段
        ent.set_field(env_ids, values, field, operation, idx_list)

    def get_stiffness(self, ctrl_ids):
        """
        获取机器人关节的刚度值
        参数:
            ctrl_ids: 控制ID（未使用）
        返回:
            机器人的刚度值
        """
        return self.get_robot("robot").get_stiffness()

    def get_dampling(self, ctrl_ids):
        """
        获取机器人关节的阻尼值（方法名拼写错误：dampling -> damping，建议修正）
        参数:
            ctrl_ids: 控制ID（未使用）
        返回:
            机器人的阻尼值
        """
        return self.get_robot("robot").get_damping()

    def render(self):
        """
        渲染场景并获取图像数据
        返回:
            形状为[720, 1280, 3]的RGB图像数组
        """
        # 将物理状态同步到USD（可视化）
        self.utils.physics_to_usd()
        # 获取视口图像数据
        img = self.app.get_viewport_image_data()
        # 重塑图像形状并提取RGB通道（丢弃Alpha通道）
        img = img.reshape([720, 1280, 4])[:, :, :3]
        return img

    def close(self):
        """关闭仿真应用，释放资源"""
        self.app.close()

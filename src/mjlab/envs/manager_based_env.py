from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# 导入实验室自定义的环境类型
from ms_lab.envs import types
# 导入MDP事件相关函数，用于重置场景到默认状态
from ms_lab.envs.mdp.events import reset_scene_to_default
# 导入各类管理器：动作管理器、事件管理器、观测管理器
from ms_lab.managers.action_manager import ActionManager
from ms_lab.managers.event_manager import EventManager
from ms_lab.managers.manager_term_config import EventTermCfg as EventTerm
from ms_lab.managers.manager_term_config import term
from ms_lab.managers.observation_manager import ObservationManager

# 场景和仿真配置相关
from ms_lab.scene.scene import SceneCfg
from ms_lab.sim import SimulationCfg

# 工具函数：随机数、日志、可视化
from ms_lab.utils import random as random_utils
from ms_lab.utils.logging import print_info
from ms_lab.viewer.debug_visualizer import DebugVisualizer
from ms_lab.viewer.viewer_config import ViewerConfig
# 后端管理器：负责管理仿真后端（如Mujoco）
from ms_lab.backend.backend_manager import BackendManager


@dataclass
class DefaultEventManagerCfg:
    """默认事件管理器配置类

  将场景重置为场景配置中指定的默认状态
  """

    # 定义重置场景到默认状态的事件项
    # term是一个工具函数，用于创建EventTerm对象
    # func: 要执行的函数（reset_scene_to_default）
    # mode: 事件触发模式（"reset"表示重置时触发）
    reset_scene_to_default: EventTerm = term(
        EventTerm,
        func=reset_scene_to_default,
        mode="reset",
    )


@dataclass(kw_only=True)  # 强制使用关键字参数初始化
class ManagerBasedEnvCfg:
    """基于管理器的环境配置类"""

    # 仿真后端类型（默认为mujoco）
    backend: str = "mujoco"
    # 物理步长的抽取因子（环境step = decimation * 物理step）
    decimation: int
    # 场景配置
    scene: SceneCfg
    # 观测配置（Any类型表示可以是任意观测配置结构）
    observations: Any
    # 动作配置（Any类型表示可以是任意动作配置结构）
    actions: Any
    # 事件配置（默认使用DefaultEventManagerCfg）
    events: Any = field(default_factory=DefaultEventManagerCfg)
    # 随机种子（None表示不设置）
    seed: int | None = None
    # 仿真配置（默认创建SimulationCfg实例）
    sim: SimulationCfg = field(default_factory=SimulationCfg)
    # 可视化器配置（默认创建ViewerConfig实例）
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


class ManagerBasedEnv:
    """基于管理器的强化学习环境核心类
  该类封装了强化学习环境的核心功能，包括：
  1. 后端仿真管理（如Mujoco）
  2. 动作、观测、事件管理器的加载和使用
  3. 环境的reset和step核心接口
  4. 随机种子管理、可视化等辅助功能
  """

    def __init__(self, cfg: ManagerBasedEnvCfg, render_mode, device: str) -> None:
        # 保存配置对象
        self.cfg = cfg

        # 设置随机种子
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            print_info("No seed set for the environment.")  # 打印无种子提示

        # 仿真步数计数器（记录物理步的总数）
        self._sim_step_counter = 0
        # 存储额外信息（如日志、奖励等）
        self.extras = {}
        # 存储观测数据的缓冲区
        self.obs_buf = {}

        # 初始化后端管理器（处理Mujoco仿真、场景加载等）
        self._backend = BackendManager(cfg.backend, render_mode=render_mode, device=device)
        # 初始化场景
        self._backend._init_scene(self.cfg.scene)
        # 初始化仿真器
        self._backend._init_sim(self.cfg.sim)

        # 如果使用CUDA，设置对应的设备
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # 打印环境基本信息
        print_info("[INFO]: Base environment:")
        print_info(f"\tEnvironment device    : {self.device}")
        print_info(f"\tEnvironment seed      : {self.cfg.seed}")
        print_info(f"\tPhysics step-size     : {self.physics_dt}")
        print_info(f"\tEnvironment step-size : {self.step_dt}")

        # 加载各类管理器（事件、动作、观测）
        self.load_managers()
        # 设置管理器的可视化器
        self.setup_manager_visualizers()

    # -------------------------- 只读属性（Property） --------------------------
    @property
    def sim(self):
        """获取仿真器实例（封装后端的sim属性）"""
        return self._backend.backend.sim

    @property
    def scene(self):
        """获取场景实例（封装后端的scene属性）"""
        return self._backend.backend.scene

    @property
    def num_envs(self) -> int:
        """获取环境数量（支持向量环境）"""
        return self._backend.num_envs

    @property
    def physics_dt(self) -> float:
        """获取物理步长（Mujoco的基础时间步）"""
        return self.cfg.sim.mujoco.timestep

    @property
    def step_dt(self) -> float:
        """获取环境步长（环境step对应的实际时间 = 物理步长 * decimation）"""
        return self.cfg.sim.mujoco.timestep * self.cfg.decimation

    @property
    def device(self) -> str:
        """获取计算设备（cpu/cuda:x）"""
        return self._backend.device

    # -------------------------- 初始化相关方法 --------------------------
    def setup_manager_visualizers(self) -> None:
        """设置管理器的可视化器
    预留方法，用于为各个管理器注册调试可视化工具
    """
        self.manager_visualizers = {}

    def load_managers(self) -> None:
        """加载并初始化所有核心管理器"""
        # 初始化事件管理器（处理场景重置、随机化等事件）
        self.event_manager = EventManager(self.cfg.events, self)
        print_info(f"[INFO] Event manager: {self.event_manager}")

        # 扩展模型字段以支持域随机化
        self._backend.expand_model_fields(self.event_manager.domain_randomization_fields)

        # 初始化动作管理器（处理动作空间、动作应用等）
        self.action_manager = ActionManager(self.cfg.actions, self)
        print_info(f"[INFO] Action Manager: {self.action_manager}")

        # 初始化观测管理器（处理观测空间、观测计算等）
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print_info(f"[INFO] Observation Manager: {self.observation_manager}")

        # 如果是基类实例且有"startup"模式的事件，则执行启动事件
        if (
                self.__class__ == ManagerBasedEnv
                and "startup" in self.event_manager.available_modes
        ):
            self.event_manager.apply(mode="startup")
            self._backend.create_graph()  # 创建仿真计算图（针对GPU加速）

    # -------------------------- MDP核心接口（Reset/Step） --------------------------
    def reset(
            self,
            *,
            seed: int | None = None,
            env_ids: torch.Tensor | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[types.VecEnvObs, dict]:
        """重置环境（强化学习MDP的重置接口）

    Args:
        seed: 重置随机种子
        env_ids: 要重置的环境ID（支持批量重置部分环境）
        options: 额外选项（未使用）

    Returns:
        tuple: (观测数据, 额外信息字典)
    """
        del options  # 显式删除未使用的参数

        # 默认重置所有环境
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # 如果指定了新种子，更新种子
        if seed is not None:
            self.seed(seed)

        # 执行环境重置逻辑
        self._reset_idx(env_ids)

        # 将数据写入仿真器
        self._backend.write_data_to_sim()
        # 执行一次前向计算（更新仿真状态）
        self._backend.forward()

        # 计算重置后的观测
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 返回观测和额外信息
        return self.obs_buf, self.extras

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[types.VecEnvObs, dict]:
        """环境步进（强化学习MDP的步进接口）

    Args:
        action: 要执行的动作张量

    Returns:
        tuple: (新的观测数据, 额外信息字典)
    """
        # 处理输入动作（转换设备、裁剪等）
        self.action_manager.process_action(action.to(self.device))

        # 执行decimation次物理步（环境步 = decimation * 物理步）
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1  # 增加物理步计数器
            self.action_manager.apply_action()  # 应用动作到仿真器
            self._backend.write_data_to_sim()  # 将数据写入仿真器
            self._backend.step()  # 执行一次物理步进
            self._backend.update(dt=self.physics_dt)  # 更新仿真状态

        # 如果有间隔触发的事件，执行这些事件
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 计算步进后的观测
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 返回新观测和额外信息
        return self.obs_buf, self.extras

    # -------------------------- 辅助方法 --------------------------
    def close(self) -> None:
        """关闭环境，释放资源"""
        self._backend.close()

    @staticmethod
    def seed(seed: int = -1) -> int:
        """设置随机种子（静态方法）

    Args:
        seed: 种子值，-1表示随机生成

    Returns:
        int: 实际使用的种子值
    """
        if seed == -1:
            seed = np.random.randint(0, 10_000)  # 随机生成种子
        print_info(f"Setting seed: {seed}")
        random_utils.seed_rng(seed)  # 设置各类随机数生成器的种子
        return seed

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        """更新所有管理器的可视化器

    Args:
        visualizer: 调试可视化器实例
    """
        for mod in self.manager_visualizers.values():
            mod.debug_vis(visualizer)

    # -------------------------- 私有方法 --------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """重置指定ID的环境（内部实现）

    Args:
        env_ids: 要重置的环境ID列表
    """
        # 调用后端重置
        self._backend.reset(env_ids)

        # 如果有重置事件，执行重置事件
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # 重置额外信息字典
        self.extras["log"] = dict()

        # 重置观测管理器并收集日志信息
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 重置动作管理器并收集日志信息
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 重置事件管理器并收集日志信息
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

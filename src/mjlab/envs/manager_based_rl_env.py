import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym  # 强化学习环境标准库
import mujoco  # Mujoco物理仿真引擎
import numpy as np
import torch
import warp as wp  # NVIDIA Warp，用于GPU加速的物理仿真

# 导入自定义类型和基础环境类
from ms_lab.envs import types
from ms_lab.envs.manager_based_env import ManagerBasedEnv, ManagerBasedEnvCfg
# 导入各类管理器：指令、课程、奖励、终止条件
from ms_lab.managers.command_manager import CommandManager, NullCommandManager
from ms_lab.managers.curriculum_manager import CurriculumManager, NullCurriculumManager
from ms_lab.managers.reward_manager import RewardManager
from ms_lab.managers.termination_manager import TerminationManager
from ms_lab.utils.logging import print_info  # 自定义日志工具


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
    """基于管理器的强化学习环境配置类
  继承自基础环境配置，新增RL特有的配置项
  """

    # 单轮episode的最大时长（秒）
    episode_length_s: float
    # 奖励函数配置
    rewards: Any
    # 终止条件配置
    terminations: Any
    # 指令管理器配置（可选，如无则使用空管理器）
    commands: Any | None = None
    # 课程学习配置（可选，用于逐步提高任务难度）
    curriculum: Any | None = None
    # 是否为有限视野MDP（finite horizon）
    is_finite_horizon: bool = False


class ManagerBasedRlEnv(ManagerBasedEnv, gym.Env):
    """基于管理器的强化学习环境类
  继承关系：
  - ManagerBasedEnv: 基础环境类（处理仿真、管理器加载等）
  - gym.Env: 符合Gymnasium标准的强化学习环境接口
  """
    # 标记为向量环境（支持多环境并行）
    is_vector_env = True
    # 环境元数据（渲染模式、版本信息等）
    metadata = {
        "render_modes": ["human", "rgb_array", None],  # 支持的渲染模式
        "mujoco_version": mujoco.__version__,  # Mujoco版本
        "warp_version": wp.config.version,  # Warp版本
    }

    # 类型注解：cfg为RL环境配置类型
    cfg: ManagerBasedRlEnvCfg

    def __init__(
            self,
            cfg: ManagerBasedRlEnvCfg,
            device: str,
            render_mode: str | None = None,
            **kwargs,
    ) -> None:
        """初始化RL环境

    Args:
        cfg: 环境配置对象
        device: 计算设备（cpu/cuda:x）
        render_mode: 渲染模式（human/rgb_array/None）
        **kwargs: 额外参数
    """
        # 通用步数计数器（记录环境step次数）
        self.common_step_counter = 0
        # 每个环境的episode长度缓冲区（记录当前episode已运行步数）
        self.episode_length_buf = torch.zeros(
            cfg.scene.num_envs, device=device, dtype=torch.long
        )

        # 调用父类初始化（基础环境配置、后端初始化等）
        super().__init__(cfg=cfg, render_mode=render_mode, device=device)
        # 保存渲染模式
        self.render_mode = render_mode

        # 如果使用Mujoco后端，初始化离线渲染器（用于rgb_array模式）
        if self.cfg.backend == "mujoco":
            from ms_lab.viewer.offscreen_renderer import OffscreenRenderer
            self._offline_renderer: OffscreenRenderer | None = None
            if self.render_mode == "rgb_array":
                # 创建离线渲染器实例
                renderer = OffscreenRenderer(
                    model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self._backend.backend._scene
                )
                renderer.initialize()  # 初始化渲染器
                self._offline_renderer = renderer

        # 设置渲染帧率（1/环境步长）
        self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore

        print_info("[INFO]: Completed setting up the environment...")

    # -------------------------- 只读属性（Property） --------------------------
    @property
    def max_episode_length_s(self) -> float:
        """获取单轮episode的最大时长（秒）"""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """计算单轮episode的最大步数（向上取整）"""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    # -------------------------- 核心方法 --------------------------
    def setup_manager_visualizers(self) -> None:
        """设置管理器的可视化器
    为指令管理器注册可视化工具（如果有激活的指令项）
    """
        self.manager_visualizers = {}
        # 如果指令管理器有激活的项，则添加到可视化器字典
        if getattr(self.command_manager, "active_terms", None):
            self.manager_visualizers["command_manager"] = self.command_manager

    def load_managers(self) -> None:
        """加载并初始化所有RL相关的管理器
    NOTE: 加载顺序很重要，需严格遵循
    """
        # 1. 初始化指令管理器（优先加载）
        if self.cfg.commands is not None:
            self.command_manager = CommandManager(self.cfg.commands, self)
        else:
            # 无指令配置时使用空管理器（空实现，避免报错）
            self.command_manager = NullCommandManager()
        print_info(f"[INFO] Command Manager: {self.command_manager}")

        # 2. 调用父类方法加载基础管理器（事件、动作、观测）
        super().load_managers()

        # 3. 初始化终止条件管理器
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print_info(f"[INFO] Termination Manager: {self.termination_manager}")

        # 4. 初始化奖励管理器
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print_info(f"[INFO] Reward Manager: {self.reward_manager}")

        # 5. 初始化课程学习管理器（可选）
        if self.cfg.curriculum is not None:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        else:
            self.curriculum_manager = NullCurriculumManager()
        print_info(f"[INFO] Curriculum Manager: {self.curriculum_manager}")

        # 6. 配置Gym环境空间（观测空间、动作空间）
        self._configure_gym_env_spaces()

        # 7. 执行启动事件（如果有）
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            self._backend.create_graph()

    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        """环境步进（核心RL接口）
    重写父类方法，增加RL特有的奖励计算、终止条件判断等逻辑

    Args:
        action: 智能体输出的动作张量

    Returns:
        VecEnvStepReturn: (观测, 奖励, 终止标志, 超时标志, 额外信息)
    """
        # 处理输入动作（设备转换、动作裁剪等）
        self.action_manager.process_action(action.to(self.device))

        # 执行decimation次物理步（环境步 = decimation * 物理步）
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1  # 物理步计数器+1
            self.action_manager.apply_action()  # 将动作应用到仿真器
            self._backend.write_data_to_sim()  # 将数据写入仿真器
            self._backend.step()  # 执行一次物理步进
            self._backend.update(dt=self.physics_dt)  # 更新仿真状态

        # 更新环境计数器
        self.episode_length_buf += 1  # 当前episode步数+1
        self.common_step_counter += 1  # 全局环境步数+1

        # 检查终止条件（包括任务完成、失败、超时等）
        self.reset_buf = self.termination_manager.compute()  # 需要重置的环境ID
        self.reset_terminated = self.termination_manager.terminated  # 任务终止标志
        self.reset_time_outs = self.termination_manager.time_outs  # 超时标志

        # 计算当前步的奖励
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # 重置那些终止/超时的环境，并记录episode信息
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)  # 执行环境重置
            self._backend.write_data_to_sim()  # 写入重置后的数据
            self._backend.forward()  # 前向计算更新状态

        # 更新指令（如目标位置、速度等）
        self.command_manager.compute(dt=self.step_dt)

        # 执行间隔事件（如域随机化、场景更新等）
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 计算当前步的观测
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 返回Gym标准的step输出
        return (
            self.obs_buf,  # 观测数据
            self.reward_buf,  # 奖励
            self.reset_terminated,  # 终止标志（任务完成/失败）
            self.reset_time_outs,  # 超时标志
            self.extras,  # 额外信息（日志、调试数据等）
        )

    def render(self) -> np.ndarray | None:
        """渲染环境（符合Gym接口）

    Returns:
        np.ndarray | None: rgb_array模式返回图像数组，human模式返回None
    """
        # 人类可视化模式或无渲染
        if self.render_mode == "human" or self.render_mode is None:
            return None
        # Mujoco后端的rgb_array模式
        elif self.render_mode == "rgb_array" and self.cfg.backend == "mujoco":
            if self._offline_renderer is None:
                raise ValueError("Offline renderer not initialized")
            # 获取可视化回调函数（如果有）
            debug_callback = (
                self.update_visualizers if hasattr(self, "update_visualizers") else None
            )
            # 执行渲染并返回图像数组
            return self._offline_renderer.render()
        # Mozi后端的rgb_array模式
        elif self.render_mode == "rgb_array" and self.cfg.backend == "mozi":
            return self._backend.render()
        # 不支持的渲染模式
        else:
            raise NotImplementedError(
                f"Render mode {self.render_mode} is not supported. "
                f"Please use: {self.metadata['render_modes']}."
            )

    def close(self) -> None:
        """关闭环境，释放资源"""
        # 关闭Mujoco离线渲染器
        if self.cfg.backend == "mujoco":  # 修复原代码的bug：self._backend是对象，不是字符串
            if self._offline_renderer is not None:
                self._offline_renderer.close()
        # 调用父类close方法释放后端资源
        super().close()

    # -------------------------- 私有方法 --------------------------
    def _configure_gym_env_spaces(self) -> None:
        """配置Gym标准的观测空间和动作空间
    基于观测/动作管理器的配置，构建符合Gym规范的空间描述
    """
        # 初始化单环境观测空间（Dict类型）
        self.single_observation_space = gym.spaces.Dict()

        # 遍历所有观测组，构建观测空间
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # 判断该组观测是否需要拼接
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            # 获取该组观测的维度
            group_dim = self.observation_manager.group_obs_dim[group_name]

            # 拼接模式：整个组作为一个Box空间
            if has_concatenated_obs:
                assert isinstance(group_dim, tuple)
                self.single_observation_space[group_name] = gym.spaces.Box(
                    low=-math.inf, high=math.inf, shape=group_dim
                )
            # 非拼接模式：每个子项作为独立的Box空间（嵌套Dict）
            else:
                assert not isinstance(group_dim, tuple)
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                for term_name, term_dim, _term_cfg in zip(
                        group_term_names, group_dim, group_term_cfgs, strict=False
                ):
                    self.single_observation_space[group_name] = gym.spaces.Dict(
                        {term_name: gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_dim)}
                    )

        # 计算动作空间维度（所有动作项维度之和）
        action_dim = sum(self.action_manager.action_term_dim)
        # 定义单环境动作空间（连续空间）
        self.single_action_space = gym.spaces.Box(
            low=-math.inf, high=math.inf, shape=(action_dim,)
        )

        # 构建向量环境的观测/动作空间（批量版本）
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """重置指定ID的环境（内部实现）
    扩展父类的重置逻辑，增加RL相关管理器的重置

    Args:
        env_ids: 要重置的环境ID列表
    """
        # 1. 执行课程学习更新（调整任务难度）
        self.curriculum_manager.compute(env_ids=env_ids)

        # 2. 重置场景元素的内部缓冲区
        self._backend.reset(env_ids)

        # 3. 执行重置事件（如场景随机化）
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # 4. 重置各类管理器（顺序敏感）
        self.extras["log"] = dict()  # 重置日志字典

        # 观测管理器重置
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 动作管理器重置
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 奖励管理器重置
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 课程学习管理器重置
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 指令管理器重置
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 事件管理器重置
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 终止条件管理器重置
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # 5. 重置episode长度缓冲区
        self.episode_length_buf[env_ids] = 0

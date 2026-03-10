"""
使用RSL-RL库训练强化学习(RL)智能体的脚本
RSL-RL: 是一个用于机器人强化学习的高性能库，专注于on-policy算法（如PPO）
"""

# 导入系统和文件操作相关模块
import os
import sys
# 数据类相关，用于配置管理
from dataclasses import asdict, dataclass
# 日期时间处理，用于日志目录命名
from datetime import datetime
# 路径处理，提供跨平台的路径操作
from pathlib import Path
# 类型注解相关
from typing import Any, cast

# 强化学习环境库
import gymnasium as gym
# 命令行参数解析库，比argparse更易用，支持dataclass
import tyro

# 本地模块导入 - RSL-RL相关配置和环境包装器
from ms_lab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# 运动跟踪任务的Runner
from ms_lab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
# 跟踪环境配置
from ms_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
# 速度控制任务的Runner
from ms_lab.tasks.velocity.rl import VelocityOnPolicyRunner
# 配置解析工具
from ms_lab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
# 工具函数：yaml保存、检查点路径获取
from ms_lab.utils.os import dump_yaml, get_checkpoint_path
# PyTorch后端配置（如精度、设备等）
from ms_lab.utils.torch import configure_torch_backends


# Weights & Biases 实验跟踪工具（暂时禁用）
# import wandb
# wandb.disabled


@dataclass(frozen=True)  # 不可变数据类，确保配置不会被意外修改
class TrainConfig:
    """
  训练配置类：集中管理所有训练相关的参数

  属性说明：
  - env: 环境配置对象
  - agent: 智能体（RSL-RL）配置对象
  - registry_name: wandb artifact的注册表名称（用于跟踪任务加载运动文件）
  - device: 训练使用的设备（如"cuda:0", "cpu"）
  - video: 是否在训练过程中录制视频
  - video_length: 每次录制视频的长度（步数）
  - video_interval: 录制视频的间隔（每多少步录制一次）
  - enable_nan_guard: 是否启用NaN检查（检测训练过程中的数值异常）
  - headless: 是否无头模式运行（无图形界面，适合服务器训练）
  """
    env: Any
    agent: RslRlOnPolicyRunnerCfg
    registry_name: str | None = None
    device: str = "cuda:0"
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    headless: bool = True


def run_train(task: str, cfg: TrainConfig) -> None:
    """
  核心训练函数：初始化环境、智能体，执行训练流程

  参数：
  - task: 任务名称（来自gym registry）
  - cfg: 训练配置对象（TrainConfig实例）
  """
    # 配置PyTorch后端（如设置默认精度、启用TF32等）
    configure_torch_backends()

    registry_name: str | None = None

    # 如果是运动跟踪任务，需要从wandb artifact加载运动文件
    if isinstance(cfg.env, TrackingEnvCfg):
        # 跟踪任务必须提供registry_name
        if not cfg.registry_name:
            raise ValueError("跟踪任务必须提供 --registry-name 参数")

        # 检查registry_name格式，如果没有指定版本，默认使用:latest
        registry_name = cast(str, cfg.registry_name)
        if ":" not in registry_name:
            registry_name = registry_name + ":latest"

        # 导入wandb并下载运动文件
        import wandb
        api = wandb.Api()
        artifact = api.artifact(registry_name)  # 获取wandb中的artifact
        # 将下载的运动文件路径配置到环境中
        cfg.env.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

    # 如果启用NaN检查，配置仿真环境的NaN guard
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard已启用，输出目录: {cfg.env.sim.nan_guard.output_dir}")

    # 配置日志目录：logs/rsl_rl/实验名/时间戳_运行名
    log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
    log_root_path = log_root_path.resolve()  # 解析为绝对路径
    print(f"[INFO] 实验日志将保存到: {log_root_path}")

    # 生成带时间戳的日志子目录
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:  # 如果指定了运行名，追加到目录名中
        log_dir += f"_{cfg.agent.run_name}"
    log_dir = log_root_path / log_dir

    # 配置渲染模式
    if cfg.video and cfg.headless:
        # 无头模式下录制视频，使用rgb_array模式（返回像素数组）
        render_mode = "rgb_array"
    elif not cfg.headless and not cfg.video:
        # 有图形界面，不录制视频，使用human模式（显示窗口）
        render_mode = "human"
    else:
        # 其他情况禁用渲染
        render_mode = None

    # 创建Gym环境实例
    env = gym.make(
        task,  # 任务名称
        cfg=cfg.env,  # 环境配置
        device=cfg.device,  # 运行设备
        render_mode=render_mode  # 渲染模式
    )

    # 处理断点续训：获取检查点路径
    resume_path = (
        get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
        if cfg.agent.resume  # 如果启用续训
        else None
    )

    # 如果启用视频录制，包装环境
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),  # 视频保存路径
            step_trigger=lambda step: step % cfg.video_interval == 0,  # 触发录制的步数
            video_length=cfg.video_length,  # 视频长度（步数）
            disable_logger=True,  # 禁用默认日志器
        )
        print("[INFO] 训练过程中将录制视频")

    # 使用RSL-RL的环境包装器包装环境（处理向量环境、动作裁剪等）
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    # 将dataclass配置转换为字典，方便保存和传递
    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    # 根据任务类型创建对应的Runner（训练器）
    if isinstance(cfg.env, TrackingEnvCfg):
        # 运动跟踪任务Runner
        runner = MotionTrackingOnPolicyRunner(
            env, agent_cfg, str(log_dir), cfg.device, registry_name
        )
    else:
        # 速度控制任务Runner
        runner = VelocityOnPolicyRunner(env, agent_cfg, str(log_dir), cfg.device)

    # 将当前代码仓库的git信息添加到日志（用于版本追踪）
    runner.add_git_repo_to_log(__file__)

    # 如果有续训路径，加载检查点
    if resume_path is not None:
        print(f"[INFO]: 从以下路径加载模型检查点: {resume_path}")
        runner.load(str(resume_path))

    # 保存配置文件到日志目录（便于复现实验）
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    # 开始训练
    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,  # 最大训练迭代次数
        init_at_random_ep_len=True  # 随机初始化episode长度（稳定训练）
    )

    # 训练结束，关闭环境
    env.close()


def main():
    """
  主函数：解析命令行参数，初始化配置，启动训练
  """
    # 第一步：解析任务名称（从gym registry中选择以"ms_lab-"开头的任务）
    task_prefix = "ms_lab-"
    chosen_task, remaining_args = tyro.cli(
        # 生成可选任务列表（仅包含ms_lab开头的任务）
        tyro.extras.literal_type_from_choices(
            [k for k in gym.registry.keys() if k.startswith(task_prefix)]
        ),
        add_help=False,  # 先不显示帮助，后续统一处理
        return_unknown_args=True,  # 返回未解析的参数，后续处理
    )
    del task_prefix  # 删除临时变量

    # 第二步：从注册表加载默认配置
    # 加载环境默认配置
    env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
    # 加载RL智能体默认配置
    agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
    # 类型检查：确保agent_cfg是预期的类型
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

    # 第三步：解析剩余命令行参数（允许覆盖默认配置）
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,  # 未解析的参数
        default=TrainConfig(env=env_cfg, agent=agent_cfg),  # 默认配置
        prog=sys.argv[0] + f" {chosen_task}",  # 命令行程序名
        config=(
            tyro.conf.AvoidSubcommands,  # 避免生成子命令
            tyro.conf.FlagConversionOff,  # 关闭布尔值自动转换为flag
        ),
    )
    # 删除临时变量，清理命名空间
    del env_cfg, agent_cfg, remaining_args

    # 启动训练
    run_train(chosen_task, args)


# 脚本入口
if __name__ == "__main__":
    main()

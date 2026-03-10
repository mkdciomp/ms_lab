"""
使用 RSL-RL 库运行强化学习(RL)智能体的脚本
主要功能：加载训练好的RL模型或使用随机/零动作智能体，在指定环境中运行并可视化
"""

import os
import sys
from dataclasses import asdict, dataclass  # 用于数据类的字典转换和定义
from pathlib import Path  # 路径处理
from typing import Literal, Optional, cast, Any  # 类型注解

# 导入自定义的ONNX导出工具
from ms_lab.tasks.velocity.rl.exporter import (
    export_velocity_policy_as_onnx,
)

# 第三方库导入
import gymnasium as gym  # 强化学习环境库
import torch  # PyTorch深度学习框架
import tyro  # 命令行参数解析工具
from rsl_rl.runners import OnPolicyRunner  # RSL-RL库的在线策略运行器

# 自定义模块导入
from ms_lab.envs import ManagerBasedRlEnvCfg  # RL环境配置基类
from ms_lab.rl import RslRlVecEnvWrapper  # RSL-RL环境包装器
from ms_lab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # 运动跟踪策略运行器
from ms_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg  # 跟踪环境配置
from ms_lab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,  # 从注册表加载配置
)
from ms_lab.utils.os import get_wandb_checkpoint_path  # 获取wandb checkpoint路径
from ms_lab.utils.torch import configure_torch_backends  # 配置PyTorch后端
from ms_lab.viewer import NativeMujocoViewer, ViserViewer, BaseViewer  # 可视化器
from ms_lab.viewer.base import EnvProtocol  # 环境协议
from ms_lab.ms_physx.viewer import MoziViewer  # Mozi物理引擎可视化器

# 定义可视化器类型选择
ViewerChoice = Literal["auto", "native", "viser"]  # auto:自动选择, native:本地, viser:网页版
ResolvedViewer = Literal["native", "viser"]  # 解析后的可视化器类型


def save_onnx(runner, path):
    """
  将训练好的策略模型保存为ONNX格式
  Args:
      runner: 策略运行器实例，包含训练好的模型
      path: checkpoint文件路径，用于推导保存位置
  """
    # 归一化器（此处暂时设为None）
    normalizer = None
    # 获取policy目录路径（去掉文件名部分）
    policy_path = path.split("model")[0]
    # 生成ONNX文件名：使用目录名作为文件名
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    # 导出模型为ONNX格式
    export_velocity_policy_as_onnx(
        runner.alg.policy,  # 策略网络
        normalizer=normalizer,  # 归一化器
        path=policy_path,  # 保存路径
        filename=filename,  # 文件名
    )


@dataclass(frozen=True)  # 不可变数据类，确保配置不被意外修改
class PlayConfig:
    """
  运行配置类，定义所有可配置的参数
  """
    env: Any  # 环境配置对象
    agent: Literal["zero", "random", "trained"] = "trained"  # 智能体类型：零动作/随机/训练好的
    registry_name: str | None = None  # wandb注册表名称
    wandb_run_path: str | None = None  # wandb运行路径，用于加载checkpoint
    checkpoint_file: str | None = None  # 本地checkpoint文件路径
    motion_file: str | None = None  # 运动文件路径（用于跟踪任务）
    num_envs: int | None = None  # 环境数量，覆盖配置文件
    device: str | None = None  # 计算设备(cuda/cpu)
    video: bool = False  # 是否录制视频
    video_length: int = 200  # 视频长度（帧数）
    video_height: int | None = None  # 视频高度
    video_width: int | None = None  # 视频宽度
    camera: int | str | None = None  # 相机ID/名称
    viewer: ViewerChoice = "auto"  # 可视化器类型


def _resolve_viewer_choice(choice: ViewerChoice) -> ResolvedViewer:
    """
  根据系统环境自动解析可视化器类型
  Args:
      choice: 用户选择的可视化器类型
  Returns:
      解析后的可视化器类型
  """
    # 如果用户指定了具体类型，直接返回
    if choice != "auto":
        return cast(ResolvedViewer, choice)

    # 检查系统是否有显示设备（Linux环境）
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    # 有显示设备用native，否则用viser（网页版）
    resolved: ResolvedViewer = "native" if has_display else "viser"
    print(f"[INFO]: Auto-selected viewer: {resolved} (display detected: {has_display})")
    return resolved


def run_play(task: str, cfg: PlayConfig):
    """
  核心运行函数：创建环境、加载智能体、运行并可视化
  Args:
      task: 任务名称（gym registry中的名称）
      cfg: 运行配置对象
  """
    # 配置PyTorch后端（优化性能、设置默认设备等）
    configure_torch_backends()

    # 确定计算设备：优先使用用户指定的，否则自动选择CUDA或CPU
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: Using device: {device}")

    # 从注册表加载环境配置
    env_cfg = cast(
        ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
    )
    # 应用用户指定的环境后端（mujoco/mozi）
    env_cfg.backend = cfg.env.backend
    # 设置环境数量
    env_cfg.scene.num_envs = cfg.env.scene.num_envs

    # 从注册表加载RL智能体配置
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
    )

    # 判断智能体模式：零动作/随机（虚拟模式） vs 训练好的（真实模式）
    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # 如果是跟踪任务（TrackingEnvCfg），需要特殊处理运动文件
    if isinstance(env_cfg, TrackingEnvCfg):
        # 虚拟模式（零动作/随机）
        if DUMMY_MODE:
            # 必须提供registry_name来获取运动文件
            if not cfg.registry_name:
                raise ValueError(
                    "Tracking tasks require `registry_name` when using dummy agents."
                )
            # 检查registry_name是否包含版本号，没有则添加":latest"
            registry_name = cast(str, cfg.registry_name)
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            # 从wandb下载运动文件
            import wandb
            api = wandb.Api()
            artifact = api.artifact(registry_name)
            env_cfg.commands.motion.motion_file = str(
                Path(artifact.download()) / "motion.npz"
            )
        # 训练好的模式
        else:
            # 使用用户指定的运动文件
            if cfg.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                env_cfg.commands.motion.motion_file = cfg.motion_file
            # 从wandb获取运动文件
            else:
                import wandb
                api = wandb.Api()
                # 检查参数合法性
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                # 从wandb run中获取运动文件
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    # 查找运动类型的artifact
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    # 设置运动文件路径
                    env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

    # 初始化日志目录和checkpoint路径
    log_dir: Optional[Path] = None
    resume_path: Optional[Path] = None

    # 训练好的模式：加载checkpoint
    if TRAINED_MODE:
        # 构建日志根路径
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        print(f"[INFO]: Loading experiment from: {log_root_path}")

        # 使用本地checkpoint文件
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        # 从wandb获取checkpoint
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))

        print(f"[INFO]: Loading checkpoint: {resume_path}")
        log_dir = resume_path.parent  # 日志目录为checkpoint的父目录

    # 覆盖环境配置参数（如果用户指定）
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    # 设置渲染模式：训练模式且需要录视频则用rgb_array，否则None
    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    # 虚拟模式下不支持录视频
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    # Mozi后端强制使用human渲染模式
    if cfg.env.backend == "physx":
        render_mode = "human"

    # 创建gym环境
    env = gym.make(task, cfg=env_cfg, device=device, render_mode=render_mode)

    # 训练模式下添加视频录制包装器
    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(Path(log_dir) / "videos" / "play"),  # 视频保存路径
            step_trigger=lambda step: step == 0,  # 仅在第0步开始录制
            video_length=cfg.video_length,  # 视频长度
            disable_logger=True,  # 禁用默认日志
        )

    # 使用RSL-RL包装器包装环境（处理动作裁剪等）
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 虚拟模式：创建零动作/随机动作策略
    if DUMMY_MODE:
        # 获取动作空间形状
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

        # 零动作策略：始终输出全零动作
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs  # 忽略观测
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()

        # 随机动作策略：输出[-1,1]范围内的随机动作
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs  # 忽略观测
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()

    # 训练模式：加载训练好的策略
    else:
        # 创建策略运行器（跟踪任务使用专用运行器）
        if isinstance(env_cfg, TrackingEnvCfg):
            runner = MotionTrackingOnPolicyRunner(
                env, asdict(agent_cfg), log_dir=str(log_dir), device=device
            )
        else:
            runner = OnPolicyRunner(
                env, asdict(agent_cfg), log_dir=str(log_dir), device=device
            )

        # 加载checkpoint
        runner.load(str(resume_path), map_location=device)
        # 保存模型为ONNX格式
        save_onnx(runner, str(resume_path))

        # 获取推理用策略（去除训练相关部分）
        policy = runner.get_inference_policy(device=device)

    # 根据后端选择可视化器并运行
    if cfg.env.backend == "mujoco":
        # 解析可视化器类型
        resolved_viewer = _resolve_viewer_choice(cfg.viewer)

        # 本地Mujoco可视化器
        if resolved_viewer == "native":
            NativeMujocoViewer(cast(EnvProtocol, env), policy).run()
        # 网页版可视化器
        elif resolved_viewer == "viser":
            ViserViewer(cast(EnvProtocol, env), policy).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    # Mozi后端使用专用可视化器
    elif cfg.env.backend == "mozi":
        MoziViewer(cast(EnvProtocol, env), policy).run()

    # 关闭环境，释放资源
    env.close()


def main():
    """
  主函数：解析命令行参数，启动运行流程
  """
    # 解析第一个参数：选择任务（以ms_lab-开头的gym任务）
    task_prefix = "ms_lab-"
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [k for k in gym.registry.keys() if k.startswith(task_prefix)]
        ),
        add_help=False,  # 先不显示帮助，后续统一处理
        return_unknown_args=True,  # 返回未解析的参数
    )

    del task_prefix  # 清理临时变量

    # 加载默认的环境和智能体配置
    env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)  # 类型检查

    # 解析剩余命令行参数（覆盖默认配置）
    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(env=env_cfg),  # 默认配置
        prog=sys.argv[0] + f" {chosen_task}",  # 程序名+任务名
        config=(
            tyro.conf.AvoidSubcommands,  # 避免子命令
            tyro.conf.FlagConversionOff,  # 关闭标志转换
        ),
    )
    del env_cfg, agent_cfg, remaining_args  # 清理临时变量

    # 启动运行流程
    run_play(chosen_task, args)


# 脚本入口
if __name__ == "__main__":
    main()

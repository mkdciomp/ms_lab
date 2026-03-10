from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, List, Tuple

# 物理材质和地面平面相关导入（用于平面地形创建）
from mozisim.core.api.materials import PhysicsMaterial
from mozisim.core.api.objects import GroundPlane

# 数值计算和张量处理库
import numpy as np
import torch

# ------------------------------ 导入上一个文件中的核心类（确保路径正确）------------------------------
# 注意：如果两个文件在同一目录下，直接导入；否则需要调整路径
from ms_lab.ms_physx.terrain import (  # 假设 terrain_generator.py 与当前文件在同一目录
    TerrainGenerator,  # 地形生成器核心类
    TerrainGeneratorCfg,  # 地形生成器配置类
    FlatTerrainCfg,  # 平坦地形配置类
    RoughTerrainCfg,  # 粗糙地形配置类
    StepTerrainCfg,  # 台阶地形配置类
    SubTerrainCfg,  # 子地形配置基类
    TerrainOutput,  # 地形输出数据结构
    HeightMapData,  # 高度图数据结构
    TerrainGeometry  # 地形几何信息结构
)


# ------------------------------ 配置类 ------------------------------
@dataclass
class TerrainImporterCfg:
    """
    地形导入器配置类
    用于配置地形导入和环境放置的相关参数
    """
    # 地形类型："generator"（程序化生成地形）/ "plane"（简单平面地形）
    terrain_type: Literal["generator", "plane"] = "plane"

    # 程序化地形生成器配置（仅 terrain_type="generator" 时需要）
    terrain_generator: Optional[TerrainGeneratorCfg] = None

    # 环境间距（平面地形或无子地形原点时使用）
    env_spacing: Optional[float] = 2.0

    # 课程式训练模式下的初始最大难度等级（行索引），None表示使用所有可用行
    max_init_terrain_level: Optional[int] = None

    # 并行环境数量（会被场景配置覆盖，如果场景配置中指定）
    num_envs: int = 1

    # 张量存储设备（cpu/cuda）
    device: str = "cpu"


# ------------------------------ 核心类 ------------------------------
class TerrainImporter:
    """
    地形导入器核心类（无MuJoCo依赖）
    核心功能：
    1. 衔接地形生成器（TerrainGenerator），获取子地形原点
    2. 计算并行环境的原点（支持课程式分配/网格布局）
    3. 管理环境与地形的对应关系（难度等级、地形类型）
    4. 支持难度动态更新（提升/降低环境对应的地形难度）
    """

    def __init__(self, cfg: TerrainImporterCfg) -> None:
        """
        初始化地形导入器
        Args:
            cfg: 地形导入器配置对象
        """
        self.cfg = cfg
        # 从配置中获取设备信息
        self.device = cfg.device

        # 校验设备可用性：如果指定CUDA但不可用则抛出异常
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA 设备 {self.device} 不可用，请使用 device='cpu'")

        # ------------------------------ 核心数据存储（均为Tensor，统一设备管理）------------------------------
        self.env_origins: Optional[torch.Tensor] = None  # (num_envs, 3) 每个环境的原点坐标 [x,y,z]
        self.terrain_origins: Optional[torch.Tensor] = None  # (num_rows, num_cols, 3) 子地形网格原点
        self.terrain_levels: Optional[torch.Tensor] = None  # (num_envs,) 每个环境对应的难度等级（行索引）
        self.terrain_types: Optional[torch.Tensor] = None  # (num_envs,) 每个环境对应的地形类型（列索引）
        self.max_terrain_level: Optional[int] = None  # 最大难度等级（num_rows - 1）
        self.terrain_sub_names: Optional[List[str]] = None  # 子地形名称列表（与列索引一一对应）

        # 根据地形类型初始化不同的地形系统
        if self.cfg.terrain_type == "generator":
            self._init_generator_terrain()
        elif self.cfg.terrain_type == "plane":
            self._init_plane_terrain()
        else:
            raise ValueError(f"未知的地形类型：{self.cfg.terrain_type}")

        # 初始化完成日志
        print(
            f"TerrainImporter 初始化完成 | 设备：{self.device} | 环境数：{self.cfg.num_envs} | 地形类型：{self.cfg.terrain_type}")

    def _init_generator_terrain(self) -> None:
        """
        初始化程序化地形（基于TerrainGenerator）
        步骤：
        1. 校验配置完整性
        2. 初始化地形生成器并生成地形数据
        3. 获取子地形原点和名称
        4. 配置课程式环境原点
        """
        # 配置校验：generator模式必须指定地形生成器配置
        if self.cfg.terrain_generator is None:
            raise ValueError("terrain_type 为 'generator' 时，必须指定 terrain_generator 配置")

        # 初始化地形生成器并生成地形数据
        terrain_generator = TerrainGenerator(
            cfg=self.cfg.terrain_generator,
            device=self.device
        )
        terrain_generator.generate()

        # 从地形生成器获取子地形原点（已在generator中移到指定设备）
        self.terrain_origins = terrain_generator.terrain_origins

        # 获取子地形名称列表（与列索引对应）
        self.terrain_sub_names = list(self.cfg.terrain_generator.sub_terrains.keys())

        # 配置课程式环境原点（按难度等级分配）
        self._configure_env_origins_curriculum()

    def _init_plane_terrain(self) -> None:
        """
        初始化平面地形（网格布局）
        步骤：
        1. 校验环境间距配置
        2. 配置网格布局的环境原点
        3. 创建物理材质和地面平面对象
        """
        # 配置校验：plane模式必须指定环境间距
        if self.cfg.env_spacing is None:
            raise ValueError("terrain_type 为 'plane' 时，必须指定 env_spacing（环境间距）")

        # 配置网格布局的环境原点
        self._configure_env_origins_grid()

        # 创建地面物理材质（摩擦力、恢复系数等）
        physics_material = PhysicsMaterial(
            prim_path="/Materials/DefaultGroundMaterial",  # 材质路径
            static_friction=0.5,  # 静摩擦系数
            dynamic_friction=0.5,  # 动摩擦系数
            restitution=0.8,  # 恢复系数（弹性）
        )

        # 地面平面尺寸（50.0对应坐标范围-25到25）
        size = 50.0

        # 地面默认颜色（灰色）
        default_color = np.array([0.1, 0.1, 0.1])

        # 实例化地面平面对象（Isaac Sim风格）
        ground_plane = GroundPlane(
            prim_path="/World/GroundPlane",  # 地面对象路径
            size=size,  # 地面尺寸
            color=default_color,  # 地面颜色
            physics_material=physics_material,  # 物理材质
            # 使用双精度浮点数确保坐标精度
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            # 隐藏渲染网格（仅保留物理碰撞）
            visible=False,
        )

    def _configure_env_origins_curriculum(self) -> None:
        """
        课程式环境原点配置（核心逻辑）
        为每个环境分配对应的子地形原点，支持难度等级控制：
        1. 确定初始最大难度等级
        2. 随机分配初始难度等级（在0~max_init_level范围内）
        3. 均匀分配地形类型
        4. 根据难度和类型映射到具体的子地形原点
        """
        # 断言确保子地形原点已初始化
        assert self.terrain_origins is not None, "子地形原点未初始化"
        # 获取子地形网格的行列数（行=难度等级，列=地形类型）
        num_rows, num_cols = self.terrain_origins.shape[:2]

        # 确定初始最大难度等级
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1  # 使用所有行
        else:
            # 取配置值和实际行数的较小值，避免越界
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # 保存最大难度等级
        self.max_terrain_level = num_rows

        # 为每个环境随机分配初始难度等级（0 ~ max_init_level）
        self.terrain_levels = torch.randint(
            low=0,
            high=max_init_level + 1,
            size=(self.cfg.num_envs,),
            device=self.device,
            dtype=torch.long
        )

        # 为每个环境均匀分配地形类型（确保每种地形类型的环境数尽可能均衡）
        self.terrain_types = self._assign_terrain_types_uniformly(num_envs=self.cfg.num_envs, num_cols=num_cols)

        # 根据难度等级和地形类型，映射到具体的子地形原点
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].contiguous()

        # 配置完成日志
        print(f"课程式环境原点配置完成 | 难度等级范围：[0, {max_init_level}] | 地形类型数：{num_cols}")

    def _configure_env_origins_grid(self) -> None:
        """
        网格布局环境原点配置（用于平面地形）
        生成近似正方形的网格布局，中心对齐世界原点：
        1. 计算网格行列数
        2. 生成网格坐标
        3. 计算每个环境的原点坐标（x/y中心对齐，z=0）
        """
        # 断言确保环境间距已配置
        assert self.cfg.env_spacing is not None, "环境间距未配置"

        # 计算网格行列数（尽量接近正方形）
        num_cols = int(np.ceil(np.sqrt(self.cfg.num_envs)))  # 列数=根号(环境数)向上取整
        num_rows = int(np.ceil(self.cfg.num_envs / num_cols))  # 行数=环境数/列数向上取整

        # 生成网格坐标矩阵（ij索引）
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device),
            torch.arange(num_cols, device=self.device),
            indexing="ij"  # ij索引模式（行优先）
        )

        # 展平网格坐标并截取前num_envs个（处理非整数网格）
        flat_ii = ii.flatten()[:self.cfg.num_envs]
        flat_jj = jj.flatten()[:self.cfg.num_envs]

        # 初始化环境原点张量（num_envs, 3）
        self.env_origins = torch.zeros((self.cfg.num_envs, 3), device=self.device, dtype=torch.float32)

        # 计算x坐标（行方向），中心对齐世界原点
        self.env_origins[:, 0] = (flat_ii - (num_rows - 1) / 2) * self.cfg.env_spacing
        # 计算y坐标（列方向），中心对齐世界原点
        self.env_origins[:, 1] = (flat_jj - (num_cols - 1) / 2) * self.cfg.env_spacing
        # z坐标固定为0（平面地形）
        self.env_origins[:, 2] = 0.0

        # 配置完成日志
        print(f"网格布局环境原点配置完成 | 网格尺寸：{num_rows}x{num_cols} | 环境间距：{self.cfg.env_spacing}m")

    def _assign_terrain_types_uniformly(self, num_envs: int, num_cols: int) -> torch.Tensor:
        """
        均匀分配地形类型（核心辅助方法）
        确保每种地形类型的环境数尽可能均衡，避免某类地形被过度使用

        Args:
            num_envs: 环境总数
            num_cols: 地形类型数（列数）

        Returns:
            地形类型索引张量 (num_envs,)
        """
        # 基础分配：每个地形类型至少分配的环境数
        base_envs_per_col = num_envs // num_cols
        # 剩余未分配的环境数
        remaining_envs = num_envs % num_cols

        # 生成地形类型索引列表
        terrain_types = []
        for col in range(num_cols):
            # 前remaining_envs个地形类型多分配一个环境
            env_count = base_envs_per_col + (1 if col < remaining_envs else 0)
            terrain_types.extend([col] * env_count)

        # 转换为Tensor并移到指定设备
        return torch.tensor(terrain_types, device=self.device, dtype=torch.long)

    def update_env_origins(
            self,
            env_ids: torch.Tensor,
            move_up: torch.Tensor,
            move_down: torch.Tensor
    ) -> None:
        """
        动态更新环境原点（课程式难度调整核心方法）
        根据训练进度提升/降低环境对应的地形难度，实现课程式学习

        Args:
            env_ids: 要更新的环境索引，shape (N,)，dtype=torch.long
            move_up: 是否提升难度（1=提升，0=不提升），shape (N,)，dtype=bool/int
            move_down: 是否降低难度（1=降低，0=不降低），shape (N,)，dtype=bool/int
        """
        # 前置校验：必要数据是否存在
        if self.terrain_origins is None or self.terrain_levels is None or self.terrain_types is None:
            print("警告：无可用的子地形数据，无法更新环境原点")
            return
        if self.max_terrain_level is None:
            print("警告：最大难度等级未初始化，无法更新环境原点")
            return

        # 设备一致性校验：确保输入张量在正确设备上
        env_ids = env_ids.to(self.device)
        move_up = move_up.to(self.device).to(torch.int)
        move_down = move_down.to(self.device).to(torch.int)

        # 维度校验：输入张量长度必须一致
        if len(env_ids) != len(move_up) or len(env_ids) != len(move_down):
            raise ValueError("env_ids、move_up、move_down 的长度必须一致")

        # 计算难度等级变化量（提升+1，降低-1，同时为1则不调整）
        level_delta = move_up - move_down
        # 更新难度等级
        self.terrain_levels[env_ids] += level_delta

        # 边界处理：确保难度等级在有效范围内
        # 1. 低于0：重置为0（最低难度）
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], min=0)
        # 2. 高于最大等级：随机分配到有效范围内（避免越界）
        overflow_mask = self.terrain_levels[env_ids] >= self.max_terrain_level
        if overflow_mask.any():
            overflow_ids = env_ids[overflow_mask]
            self.terrain_levels[overflow_ids] = torch.randint(
                low=0,
                high=self.max_terrain_level,
                size=(overflow_ids.shape[0],),
                device=self.device,
                dtype=torch.long
            )

        # 根据新的难度等级和地形类型更新环境原点
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

        # 更新日志（可注释掉以提升性能）
        # print(f"更新完成 | 处理环境数：{len(env_ids)} | 提升难度：{move_up.sum().item()} 个 | 降低难度：{move_down.sum().item()} 个")

    # ------------------------------ 辅助方法 ------------------------------
    def get_env_origin(self, env_id: int) -> Optional[np.ndarray]:
        """
        获取单个环境的原点坐标（转换为numpy数组，方便外部使用）

        Args:
            env_id: 环境索引

        Returns:
            环境原点坐标 [x,y,z] 或 None（失败时）
        """
        # 数据校验
        if self.env_origins is None:
            print("警告：环境原点未初始化")
            return None
        if env_id < 0 or env_id >= self.cfg.num_envs:
            print(f"警告：环境索引 {env_id} 超出范围（0~{self.cfg.num_envs - 1}）")
            return None

        # 转换为numpy数组并返回
        return self.env_origins[env_id].cpu().numpy()

    def get_terrain_info(self, env_id: int) -> Optional[Dict]:
        """
        获取单个环境的地形详细信息（调试/日志用）

        Args:
            env_id: 环境索引

        Returns:
            地形信息字典或None（失败时）
        """
        # 索引范围校验
        if env_id < 0 or env_id >= self.cfg.num_envs:
            print(f"警告：环境索引 {env_id} 超出范围（0~{self.cfg.num_envs - 1}）")
            return None

        # 基础信息
        info = {
            "env_id": env_id,
            "terrain_type": self.cfg.terrain_type,
            "env_origin": self.get_env_origin(env_id),
            "max_terrain_level": self.max_terrain_level
        }

        # 程序化地形额外信息
        if self.cfg.terrain_type == "generator":
            if self.terrain_levels is None or self.terrain_types is None or self.terrain_sub_names is None:
                return info
            # 获取具体的难度等级和地形类型信息
            terrain_level = self.terrain_levels[env_id].item()
            terrain_type_idx = self.terrain_types[env_id].item()
            terrain_name = self.terrain_sub_names[terrain_type_idx] if terrain_type_idx < len(
                self.terrain_sub_names) else "unknown"
            # 更新信息字典
            info.update({
                "terrain_level": terrain_level,  # 难度等级
                "terrain_type_idx": terrain_type_idx,  # 地形类型索引
                "terrain_name": terrain_name  # 地形名称
            })

        return info

    def get_all_env_origins(self) -> Optional[np.ndarray]:
        """
        获取所有环境的原点坐标（批量导出用）

        Returns:
            所有环境原点数组 (num_envs, 3) 或 None（失败时）
        """
        if self.env_origins is None:
            print("警告：环境原点未初始化")
            return None
        return self.env_origins.cpu().numpy()

    def get_terrain_summary(self) -> Dict:
        """
        获取地形系统整体摘要信息（统计/日志用）

        Returns:
            地形摘要信息字典
        """
        # 基础摘要信息
        summary = {
            "terrain_type": self.cfg.terrain_type,
            "num_envs": self.cfg.num_envs,
            "device": self.device,
            "env_spacing": self.cfg.env_spacing,
            "max_terrain_level": self.max_terrain_level,
            "terrain_sub_names": self.terrain_sub_names
        }

        # 程序化地形额外摘要信息
        if self.terrain_origins is not None:
            num_rows, num_cols = self.terrain_origins.shape[:2]
            summary.update({
                "sub_terrain_grid": f"{num_rows}x{num_cols}",  # 子地形网格尺寸
                "num_difficulty_levels": num_rows,  # 难度等级数
                "num_terrain_types": num_cols  # 地形类型数
            })

        return summary


# ------------------------------ 完整使用示例 ------------------------------
if __name__ == "__main__":
    # 1. 配置子地形（与 terrain_generator.py 一致）
    sub_terrains: Dict[str, SubTerrainCfg] = {
        "flat": FlatTerrainCfg(
            proportion=0.2,  # 占比20%
            resolution=(128, 128),  # 高度图分辨率
            height=0.0  # 基础高度
        ),
        "rough": RoughTerrainCfg(
            proportion=0.5,  # 占比50%
            resolution=(128, 128),  # 高度图分辨率
            base_height=0.0,  # 基础高度
            max_height_diff=1.0,  # 最大高度差
            noise_scale=5.0  # 噪声尺度
        ),
        "step": StepTerrainCfg(
            proportion=0.3,  # 占比30%
            resolution=(128, 128),  # 高度图分辨率
            min_step_height=0.1,  # 最小台阶高度
            max_step_height=0.6,  # 最大台阶高度
            min_step_count=2,  # 最小台阶数
            max_step_count=8  # 最大台阶数
        )
    }

    # 2. 配置地形生成器
    generator_cfg = TerrainGeneratorCfg(
        seed=42,  # 随机种子（确保可复现）
        curriculum=True,  # 启用课程式训练
        size=(8.0, 8.0),  # 每个子地形尺寸
        border_width=1.0,  # 边界宽度
        border_height=1.2,  # 边界高度
        num_rows=4,  # 4个难度等级（行）
        num_cols=3,  # 3种地形类型（列）
        color_scheme="height",  # 高度着色方案
        sub_terrains=sub_terrains,  # 子地形配置
        difficulty_range=(0.1, 0.9),  # 难度范围
        add_border=True  # 添加边界
    )

    # 3. 配置地形导入器
    importer_cfg = TerrainImporterCfg(
        terrain_type="generator",  # 使用程序化地形
        terrain_generator=generator_cfg,  # 地形生成器配置
        num_envs=10,  # 10个并行环境
        max_init_terrain_level=2,  # 初始最大难度等级（0~2）
        env_spacing=2.0,  # 环境间距
        device="cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    )

    # 4. 创建地形导入器实例
    importer = TerrainImporter(cfg=importer_cfg)

    # 5. 查看地形系统整体摘要
    print("\n=== 地形整体摘要 ===")
    summary = importer.get_terrain_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    # 6. 查看前5个环境的详细地形信息
    print("\n=== 前5个环境的地形信息 ===")
    for env_id in range(5):
        info = importer.get_terrain_info(env_id)
        if info:
            print(f"\n环境 {info['env_id']}:")
            for key, value in info.items():
                if key == "env_origin":
                    print(f"  {key}: {value.round(3)}")  # 坐标保留3位小数
                else:
                    print(f"  {key}: {value}")

    # 7. 动态更新环境难度示例
    print("\n=== 动态更新环境难度 ===")
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)  # 要更新的环境ID
    move_up = torch.tensor([1, 1, 0], dtype=torch.bool)  # 提升环境0、1的难度
    move_down = torch.tensor([0, 0, 1], dtype=torch.bool)  # 降低环境2的难度
    importer.update_env_origins(env_ids, move_up, move_down)

    # 8. 查看更新后的环境信息
    print("\n=== 更新后环境0、1、2的信息 ===")
    for env_id in [0, 1, 2]:
        info = importer.get_terrain_info(env_id)
        if info:
            print(f"\n环境 {info['env_id']}:")
            print(f"  难度等级: {info['terrain_level']}")
            print(f"  地形名称: {info['terrain_name']}")
            print(f"  原点坐标: {info['env_origin'].round(3)}")

    # 9. 导出所有环境原点到文件
    all_origins = importer.get_all_env_origins()
    if all_origins is not None:
        np.save("all_env_origins.npy", all_origins)
        print("\n所有环境原点已保存到 all_env_origins.npy")

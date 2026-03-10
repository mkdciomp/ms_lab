from __future__ import annotations
import mujoco
from ms_lab.ms_physx.hf2mesh import generate_mesh_from_height_field  # 高度场转网格函数
import numpy as np

# 导入高度场地形配置类
from ms_lab.ms_physx.hf import *

# 地形类型到配置类的映射字典
# 键：地形名称，值：对应的高度场配置类
TERRAIN_DICT = {
    "flat": HfFlatTerrainCfg,                      # 平坦地形
    "pyramid_stairs": HfPyramidStairsTerrainCfg,   # 金字塔台阶地形
    "pyramid_stairs_inv": HfPyramidStairsTerrainCfg, # 倒置金字塔台阶地形
    "hf_pyramid_slope": HfPyramidSlopedTerrainCfg, # 金字塔斜坡地形
    "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg, # 倒置金字塔斜坡地形
    "random_rough": HfRandomUniformTerrainCfg,     # 随机粗糙地形
    "wave_terrain": HfWaveTerrainCfg,              # 波浪地形
}

# 导入mozisim相关模块（用于网格和材质定义）
from mozisim.core.api.mesh.mesh import FixedMesh, DynamicMesh  # 固定/动态网格类
from mozisim.core.api.materials import PhysicsMaterial        # 物理材质类
from mozisim.utils.mesh_utils import color_by_numerical_feature  # 数值特征着色函数

# 导入基础库
import abc                     # 抽象基类
import time                    # 计时
from dataclasses import dataclass  # 数据类装饰器
from typing import Literal, Dict, List, Optional, Tuple  # 类型注解
import torch                   # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数库

# ------------------------------ 全局常量 ------------------------------
_DARK_GRAY = (0.2, 0.2, 0.2, 1.0)  # 地形边界默认颜色（深灰色，RGBA）


# ------------------------------ 基础数据结构 ------------------------------
@dataclass
class HeightMapData:
    """
    高度图核心数据结构（无MuJoCo依赖）
    存储高度图的数值数据、物理尺寸、空间位置等信息
    """
    data: torch.Tensor  # 高度值矩阵，形状为 (resolution_y, resolution_x)
    origin: torch.Tensor  # 高度图左下角在世界坐标系中的位置 (x, y, z)
    size: Tuple[float, float]  # 高度图对应的物理尺寸 (width, depth) -> (x方向长度, y方向长度)
    resolution: Tuple[int, int]  # 高度图分辨率 (resolution_y, resolution_x)
    min_height: float  # 高度图最小高度（用于着色和归一化）
    max_height: float  # 高度图最大高度（用于着色和归一化）

    def __post_init__(self):
        """
        数据类初始化后自动计算最小/最大高度
        避免手动传入，保证数据一致性
        """
        self.min_height = float(self.data.min().item())
        self.max_height = float(self.data.max().item())


@dataclass
class TerrainGeometry:
    """
    地形几何数据容器
    包含高度图数据和对应的颜色信息
    """
    height_map: Optional[HeightMapData] = None  # 高度图数据（核心）
    color: Optional[Tuple[float, float, float, float]] = None  # RGBA颜色（None则使用默认色）



# ------------------------------ 配置类 ------------------------------
@dataclass
class SubTerrainCfg(abc.ABC):
    """
    子地形配置抽象类（可扩展不同地形类型）
    所有具体地形类型都需继承此类并实现function方法
    """
    proportion: float = 1.0  # 随机模式下该地形被选中的概率权重
    size: Tuple[float, float] = (10.0, 10.0)  # 子地形物理尺寸 (width, depth)，单位米
    resolution: Tuple[int, int] = (128, 128)  # 高度图分辨率（越高越精细，128x128为平衡值）

    @abc.abstractmethod
    def function(
            self, difficulty: float, rng: torch.Generator, device: str
    ) -> TerrainOutput:
        """
        生成子地形数据的核心方法（必须由子类实现）
        Args:
            difficulty: 难度系数 [0, 1]，控制地形复杂度（0最简单，1最复杂）
            rng: torch随机数生成器（保证可复现性）
            device: 数据存储设备（cpu/cuda）
        Returns:
            TerrainOutput: 包含子地形完整数据的对象
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class TerrainGeneratorCfg:
    """
    地形生成器全局配置类
    控制整个地形网格的生成规则和参数
    """
    seed: Optional[int] = None  # 随机种子（None则使用随机种子，用于复现实验）
    curriculum: bool = False  # 难度模式：True=课程式（行索引越大难度越高）| False=随机难度
    size: Tuple[float, float]  # 单个子地形的物理尺寸 (width, depth)，单位米
    border_width: float = 0.0  # 地形整体边界宽度（0则无边界），单位米
    border_height: float = 1.0  # 边界高度，单位米
    num_rows: int = 1  # 地形网格行数（课程式难度的核心维度）
    num_cols: int = 1  # 地形网格列数
    color_scheme: Literal["height", "random", "none"] = "height"  # 颜色方案：按高度/随机/无颜色
    sub_terrains: Dict[str, SubTerrainCfg]  # 子地形配置字典（key: 地形名称, value: 配置对象）
    difficulty_range: Tuple[float, float] = (0.0, 1.0)  # 全局难度范围 [min, max]
    add_border: bool = True  # 是否生成地形边界（防止机器人掉落）


# ------------------------------ 工具函数 ------------------------------
def gaussian_filter_torch(input_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Torch实现的高斯平滑滤波器（替代scipy.ndimage.gaussian_filter）
    用于平滑地形高度图，使地形过渡更自然
    Args:
        input_tensor: 输入张量，形状为 (H, W)
        sigma: 高斯核标准差（值越大越平滑）
    Returns:
        torch.Tensor: 平滑后的张量，形状与输入一致
    """
    if sigma <= 0:  # 标准差≤0时直接返回原张量
        return input_tensor.clone()

    # 计算高斯核大小（3σ原则：核大小覆盖99.7%的高斯分布）
    sigma_tensor = torch.tensor(sigma, device=input_tensor.device, dtype=torch.float32)
    kernel_size = int(2 * torch.ceil(3 * sigma_tensor) + 1)
    if kernel_size % 2 == 0:  # 保证核大小为奇数
        kernel_size += 1

    # 生成1D高斯核
    x = torch.arange(kernel_size, device=input_tensor.device, dtype=torch.float32) - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma_tensor.pow(2)))  # 高斯公式
    gaussian_1d /= gaussian_1d.sum()  # 归一化

    # 生成2D高斯核（外积）
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)

    # 扩展维度以适应卷积（PyTorch卷积要求4D输入：[batch, channel, H, W]）
    input_4d = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel_4d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # 填充并卷积（padding保证输出尺寸与输入一致）
    padding = kernel_size // 2
    smoothed = F.conv2d(
        input_4d, kernel_4d, padding=padding, groups=1
    )

    # 恢复为2D张量
    return smoothed.squeeze(0).squeeze(0)


def make_border_data(
        border_size: Tuple[float, float],
        inner_size: Tuple[float, float],
        height: float,
        origin: torch.Tensor,
        device: str
) -> List[TerrainGeometry]:
    """
    生成地形边界的高度图数据（替代原MuJoCo的几何边界）
    生成上下左右四个边界，防止机器人走出地形区域
    Args:
        border_size: 边界整体尺寸 (width, depth)
        inner_size: 内部地形区域尺寸 (width, depth)
        height: 边界高度（高于内部地形）
        origin: 边界中心原点（Tensor）
        device: 数据存储设备（cpu/cuda）
    Returns:
        List[TerrainGeometry]: 四个边界的地形几何数据列表
    """
    border_geoms = []
    # 计算半宽/半高，方便坐标计算
    half_bw, half_bh = border_size[0] / 2, border_size[1] / 2
    half_iw, half_ih = inner_size[0] / 2, inner_size[1] / 2
    border_resolution = (32, 128)  # 边界高度图分辨率（低分辨率即可，节省计算）

    # 1. 上边界（y方向正方向）
    upper_origin = origin + torch.tensor([0.0, half_ih, 0.0], device=device, dtype=torch.float32)
    upper_height_data = torch.full(border_resolution, height, dtype=torch.float32, device=device)
    border_geoms.append(TerrainGeometry(
        height_map=HeightMapData(
            data=upper_height_data,
            origin=upper_origin,
            size=(inner_size[0], border_size[1] - inner_size[1]),
            resolution=border_resolution
        ),
        color=_DARK_GRAY  # 使用默认深灰色
    ))

    # 2. 下边界（y方向负方向）
    lower_origin = origin + torch.tensor([0.0, -half_ih, 0.0], device=device, dtype=torch.float32)
    lower_height_data = torch.full(border_resolution, height, dtype=torch.float32, device=device)
    border_geoms.append(TerrainGeometry(
        height_map=HeightMapData(
            data=lower_height_data,
            origin=lower_origin,
            size=(inner_size[0], border_size[1] - inner_size[1]),
            resolution=border_resolution
        ),
        color=_DARK_GRAY
    ))

    # 3. 左边界（x方向负方向）
    left_origin = origin + torch.tensor([-half_iw, 0.0, 0.0], device=device, dtype=torch.float32)
    left_height_data = torch.full((128, 32), height, dtype=torch.float32, device=device)
    border_geoms.append(TerrainGeometry(
        height_map=HeightMapData(
            data=left_height_data,
            origin=left_origin,
            size=(border_size[0] - inner_size[0], inner_size[1]),
            resolution=(128, 32)
        ),
        color=_DARK_GRAY
    ))

    # 4. 右边界（x方向正方向）
    right_origin = origin + torch.tensor([half_iw, 0.0, 0.0], device=device, dtype=torch.float32)
    right_height_data = torch.full((128, 32), height, dtype=torch.float32, device=device)
    border_geoms.append(TerrainGeometry(
        height_map=HeightMapData(
            data=right_height_data,
            origin=right_origin,
            size=(border_size[0] - inner_size[0], inner_size[1]),
            resolution=(128, 32)
        ),
        color=_DARK_GRAY
    ))

    return border_geoms


# ------------------------------ 核心生成器类 ------------------------------
class TerrainGenerator:
    """
    无MuJoCo依赖的地形生成器核心类
    专注于高度图和网格数据生成（全Torch实现，兼容旧PyTorch版本）
    支持课程式难度和随机难度两种模式
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
        """
        初始化地形生成器
        Args:
            cfg: 全局配置对象
            device: 数据存储设备（cpu/cuda）
        """
        # 设备检查：指定CUDA但不可用时抛出错误
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请使用 device='cpu'")

        # 输入参数校验（保证配置合法性）
        if len(cfg.sub_terrains) == 0:
            raise ValueError("配置错误：sub_terrains 不能为空，至少需要一个子地形类型")
        if cfg.num_rows < 1 or cfg.num_cols < 1:
            raise ValueError("配置错误：num_rows 和 num_cols 必须大于等于1")
        if cfg.difficulty_range[0] > cfg.difficulty_range[1]:
            raise ValueError("配置错误：difficulty_range 第一个值不能大于第二个值")

        # 保存配置和设备信息
        self.cfg = cfg
        self.color_index = False  # 颜色交替标志（用于棋盘格着色）
        self.device = device

        # 统一所有子地形的尺寸和分辨率（由全局配置控制，保证一致性）
        for sub_cfg in self.cfg.sub_terrains.values():
            sub_cfg.size = self.cfg.size
            sub_cfg.resolution = (128, 128)  # 统一分辨率，可根据需求调整

        # 初始化随机数生成器（兼容旧PyTorch版本）
        # 无种子时生成随机种子，有种子时使用指定种子
        seed = self.cfg.seed if self.cfg.seed is not None else torch.randint(0, 10000, (1,), device=device).item()
        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)
        print(f"地形生成器初始化完成 | 随机种子：{seed} | 设备：{device}")

        # 核心数据存储初始化
        # 每个子地形的spawn原点，形状：(num_rows, num_cols, 3)
        self.terrain_origins: torch.Tensor = torch.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, 3),
            device=device, dtype=torch.float32
        )
        # 二维网格的地形数据，结构：[行][列] -> TerrainOutput
        self.terrain_data: List[List[TerrainOutput]] = []
        # 边界数据（None表示未生成）
        self.border_data: Optional[List[TerrainGeometry]] = None
        # 全局地形高度范围（用于统一着色和碰撞检测）
        self.global_height_range: Tuple[float, float] = (0.0, 0.0)

    def generate(self) -> None:
        """
        生成完整的地形数据（主入口方法）
        执行流程：清空历史数据 → 生成子地形网格 → 生成边界（可选）→ 计算全局高度范围
        """
        # 清空历史数据，防止累积
        self.terrain_data = []
        self.terrain_origins.zero_()

        # 根据难度模式生成子地形网格
        if self.cfg.curriculum:
            tic = time.perf_counter()  # 开始计时
            self._generate_curriculum_terrains()  # 课程式难度
            toc = time.perf_counter()  # 结束计时
            print(
                f"课程式地形网格生成完成 | 行数：{self.cfg.num_rows} | 列数：{self.cfg.num_cols} | 耗时：{toc - tic:.4f}秒")
        else:
            tic = time.perf_counter()
            self._generate_random_terrains()  # 随机难度
            toc = time.perf_counter()
            print(
                f"随机地形网格生成完成 | 行数：{self.cfg.num_rows} | 列数：{self.cfg.num_cols} | 耗时：{toc - tic:.4f}秒")

        # 生成边界（如果启用且边界宽度>0）
        # if self.cfg.border_width > 0:
        #     tic = time.perf_counter()
        #     self._generate_border_data()
        #     toc = time.perf_counter()
        #     print(
        #         f"边界生成完成 | 宽度：{self.cfg.border_width} | 高度：{self.cfg.border_height} | 耗时：{toc - tic:.4f}秒")

    def _generate_random_terrains(self) -> None:
        """
        生成随机难度的地形网格
        每个子地形随机选择类型（按权重）和难度（在指定范围内）
        """
        # 获取子地形配置和名称列表
        sub_names = list(self.cfg.sub_terrains.keys())
        sub_cfgs_mozi = {}

        # 将用户配置转换为mozisim兼容的高度场配置
        for name, cfg in self.cfg.sub_terrains.items():
            # 创建对应类型的高度场配置对象
            cfg_mozi = TERRAIN_DICT[name]()
            # 获取配置属性字典
            mozi_attr = cfg_mozi.__dict__
            src_attr = cfg.__dict__

            # 将用户配置的属性复制到mozisim配置中
            for key in src_attr.keys():
                if key in mozi_attr.keys():
                    setattr(cfg_mozi, key, src_attr[key])
            # 处理倒置地形标记
            if name.split("_")[-1] == "inv":
                cfg_mozi.inverted = True
            # 设置设备
            cfg_mozi.device = self.device
            sub_cfgs_mozi[name] = cfg_mozi

        # 转换为列表方便索引
        sub_cfgs = list(sub_cfgs_mozi.values())
        sub_names = list(self.cfg.sub_terrains.keys())

        # 归一化子地形选择权重（保证权重和为1）
        proportions = torch.tensor(
            [cfg.proportion for cfg in sub_cfgs],
            dtype=torch.float32, device=self.device
        )
        proportions /= proportions.sum()

        # 遍历所有网格位置生成地形
        total_terrains = self.cfg.num_rows * self.cfg.num_cols
        for row in range(self.cfg.num_rows):
            current_row = []  # 存储当前行的地形数据
            for col in range(self.cfg.num_cols):
                # 随机选择子地形类型（按权重抽样）
                sub_idx = torch.multinomial(
                    proportions, num_samples=1, generator=self.torch_rng
                ).item()
                selected_sub_cfg = sub_cfgs[sub_idx]
                selected_sub_name = sub_names[sub_idx]

                # 随机生成难度值（在指定范围内）
                difficulty = self._generate_uniform(
                    low=self.cfg.difficulty_range[0],
                    high=self.cfg.difficulty_range[1]
                ).item()

                # 计算子地形在世界坐标系中的位置
                world_pos = self._get_sub_terrain_position(row, col)

                # 生成子地形数据
                terrain_output = self._create_terrain_data(
                    world_pos, difficulty, selected_sub_cfg, row, col
                )

                # 存储生成的地形数据
                self.terrain_origins[row, col] = terrain_output
                current_row.append(terrain_output)

                # 进度日志（每10个地形或最后一个地形输出）
                terrain_idx = row * self.cfg.num_cols + col + 1
                if terrain_idx % 10 == 0 or terrain_idx == total_terrains:
                    print(
                        f"已生成 {terrain_idx}/{total_terrains} 个地形 | 类型：{selected_sub_name} | 难度：{difficulty:.3f}")

            self.terrain_data.append(current_row)

    def _generate_curriculum_terrains(self) -> None:
        """
        生成课程式难度的地形网格
        核心规则：
        1. 行索引越大，难度越高（基础难度+随机扰动）
        2. 每列固定一种地形类型（按权重分配）
        """
        # 获取子地形配置和名称列表
        sub_names = list(self.cfg.sub_terrains.keys())
        sub_cfgs_mozi = {}

        # 将用户配置转换为mozisim兼容的高度场配置
        for name, cfg in self.cfg.sub_terrains.items():
            cfg_mozi = TERRAIN_DICT[name]()
            mozi_attr = cfg_mozi.__dict__
            src_attr = cfg.__dict__

            # 复制配置属性
            for key in src_attr.keys():
                if key in mozi_attr.keys():
                    setattr(cfg_mozi, key, src_attr[key])
            # 处理倒置地形
            if name.split("_")[-1] == "inv":
                cfg_mozi.inverted = True
            cfg_mozi.device = self.device
            sub_cfgs_mozi[name] = cfg_mozi

        # 转换为列表
        sub_cfgs = list(sub_cfgs_mozi.values())
        sub_names = list(self.cfg.sub_terrains.keys())

        # 归一化子地形权重并计算累积权重
        proportions = torch.tensor(
            [cfg.proportion for cfg in sub_cfgs],
            dtype=torch.float32, device=self.device
        )
        proportions /= proportions.sum()
        cumsum_props = proportions.cumsum(dim=0)

        # 为每列分配固定的子地形类型（按列索引比例分配）
        col_sub_indices = []
        num_cols = self.cfg.num_cols
        for col in range(num_cols):
            # 计算列对应的比例阈值
            col_ratio = torch.tensor(
                (col + 1) / num_cols,
                device=self.device, dtype=torch.float32
            )
            # 查找第一个大于等于阈值的地形类型索引
            mask = cumsum_props >= col_ratio

            if mask.sum() == 0:
                # 无匹配时使用默认索引0
                sub_idx = 0
            else:
                sub_idx = list(torch.where(mask))[0][0].item()

            col_sub_indices.append(sub_idx)

        # 遍历所有网格位置生成地形
        lower_diff, upper_diff = self.cfg.difficulty_range
        num_rows = self.cfg.num_rows
        for row in range(num_rows):
            current_row = []  # 存储当前行的地形数据
            for col in range(num_cols):
                # 计算当前行的难度（随行数递增，加入小随机扰动）
                rand_perturb = self._generate_uniform(low=0.0, high=0.5).item()
                difficulty = (row + rand_perturb) / num_rows
                # 难度裁剪到[0,1]范围，并映射到全局难度范围
                difficulty = lower_diff + (upper_diff - lower_diff) * torch.clip(
                    torch.tensor(difficulty, device=self.device), 0.0, 1.0
                ).item()

                # 获取当前列的固定地形类型
                sub_idx = col_sub_indices[col]
                selected_sub_cfg = sub_cfgs[sub_idx]
                selected_sub_name = sub_names[sub_idx]
                print(f"第{row}行第{col}列地形类型：{selected_sub_name}")

                # 计算子地形在世界坐标系中的位置
                world_pos = self._get_sub_terrain_position(row, col)

                # 生成子地形数据
                terrain_output = self._create_terrain_data(
                    world_pos, difficulty, selected_sub_cfg, row, col
                )

                # 存储地形数据
                self.terrain_origins[row, col] = terrain_output
                current_row.append(terrain_output)

            self.terrain_data.append(current_row)
            # 输出当前行的难度范围
            row_diff_min = lower_diff + (upper_diff - lower_diff) * row / num_rows
            row_diff_max = lower_diff + (upper_diff - lower_diff) * (row + 0.5) / num_rows
            print(
                f"已生成第 {row + 1}/{num_rows} 行课程地形 | 难度范围：[{row_diff_min:.3f}, {row_diff_max:.3f}]")

    def _get_sub_terrain_position(self, row: int, col: int) -> torch.Tensor:
        """
        计算子地形左下角在世界坐标系中的位置
        核心规则：整个地形网格中心对齐世界原点 (0,0,0)
        Args:
            row: 行索引
            col: 列索引
        Returns:
            torch.Tensor: 世界坐标位置 (x, y, z)
        """
        # 子地形尺寸
        terrain_width, terrain_depth = self.cfg.size

        # 子地形在网格中的相对位置（左下角为基准）
        rel_x = row * terrain_width
        rel_y = col * terrain_depth

        # 计算网格整体尺寸
        grid_total_width = self.cfg.num_rows * terrain_width
        grid_total_depth = self.cfg.num_cols * terrain_depth

        # 计算网格整体偏移量（使网格中心对齐世界原点）
        grid_offset_x = - (grid_total_width - terrain_width) / 2
        grid_offset_y = - (grid_total_depth - terrain_depth) / 2

        # 最终世界位置（z轴为0，即地形底部高度）
        return torch.tensor(
            [grid_offset_x + rel_x, grid_offset_y + rel_y, 0.0],
            device=self.device, dtype=torch.float32
        )

    def _create_terrain_data(
            self,
            world_position: torch.Tensor,
            difficulty: float,
            sub_cfg: SubTerrainCfg,
            row: int,
            col: int,
    ) -> TerrainOutput:
        """
        生成单个子地形的完整数据，并应用世界位置偏移
        核心流程：高度场生成 → 网格转换 → 物理网格创建 → 坐标调整
        Args:
            world_position: 子地形左下角在世界坐标系中的位置（Tensor）
            difficulty: 难度系数
            sub_cfg: 子地形配置对象
            row: 行索引（用于命名）
            col: 列索引（用于命名）
        Returns:
            TerrainOutput: 包含世界坐标偏移后的子地形数据
        """
        # 从高度场生成网格数据
        mesh, origin = generate_mesh_from_height_field(difficulty, sub_cfg)
        # 可选：导出网格为OBJ文件（用于调试）
        # mesh.export(file_obj=f"./mesh/mesh_{row}_{col}.obj", file_type="obj")

        # 计算位置偏移（边界宽度补偿）
        bias = origin
        bias[:2] = sub_cfg.border_width / 2  # 边界宽度补偿

        # 计算子地形的世界位置
        world_position_sub_r = bias + world_position
        # 计算物理引擎中的世界位置（中心对齐）
        engine_world_position = world_position.cpu().numpy() - np.array([
            (sub_cfg.size[0] + sub_cfg.border_width) / 2,
            (sub_cfg.size[0] + sub_cfg.border_width) / 2,
            0.0,
        ])

        # 1. 提取网格顶点坐标（用于创建物理网格）
        mesh_points = mesh.vertices
        mesh_points = np.array(mesh_points)

        # 2. 提取面拓扑信息
        # 面顶点索引：将二维的faces（N,3）展平为一维数组
        mesh_face_indices = mesh.faces.flatten()
        mesh_face_indices = np.array(mesh_face_indices)
        # 面顶点数量：每个三角面有3个顶点
        mesh_face_counts = [3] * len(mesh.faces)
        mesh_face_counts = np.array(mesh_face_counts)

        # 实例化物理材质（地面材质属性）
        physics_material = PhysicsMaterial(
            prim_path="/Materials/DefaultGroundMaterial",
            static_friction=0.5,     # 静摩擦系数
            dynamic_friction=0.5,    # 动摩擦系数
            restitution=0.8,         # 恢复系数（弹性）
        )

        # 设置网格颜色（棋盘格模式）
        if (row + col) % 2 == 0:
            default_color = np.array([0.1, 0.1, 0.1])  # 深灰
        else:
            default_color = np.array([0.4, 0.4, 0.4])  # 浅灰
            self.color_index = not self.color_index

        # 创建固定物理网格（添加到场景中）
        FixedMesh(
            prim_path=f"/World/GroundPlane_{row}_{col}",  # 唯一路径
            points=mesh_points.tolist(),                  # 顶点列表
            face_indices=mesh_face_indices.tolist(),      # 面索引列表
            face_counts=mesh_face_counts.tolist(),        # 面顶点数列表
            position=engine_world_position.tolist(),      # 世界位置
            color=default_color,                          # 网格颜色
            physics_material=physics_material,            # 物理材质
            collision_approximation="sdf",                # 碰撞近似方式（SDF更精确）
        )

        # 根据高度特征着色（可选）
        color_by_numerical_feature("terrain", mesh_points, f"/World/GroundPlane_{row}_{col}")

        return world_position_sub_r

    def _generate_border_data(self) -> None:
        """
        生成地形边界的高度图数据
        调用make_border_data函数，计算边界尺寸并生成四个边界
        """
        # 计算内部地形和边界的尺寸
        inner_size = (
            self.cfg.num_rows * self.cfg.size[0],
            self.cfg.num_cols * self.cfg.size[1]
        )
        border_size = (
            inner_size[0] + 2 * self.cfg.border_width,
            inner_size[1] + 2 * self.cfg.border_width
        )

        # 生成边界数据（中心对齐世界原点）
        self.border_data = make_border_data(
            border_size=border_size,
            inner_size=inner_size,
            height=self.cfg.border_height,
            origin=torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32),
            device=self.device
        )

    def _compute_global_height_range(self) -> None:
        """
        计算全局地形的高度范围（所有子地形+边界）
        用于统一着色、碰撞检测和机器人初始化高度设置
        """
        min_heights = []  # 存储所有地形的最小高度
        max_heights = []  # 存储所有地形的最大高度

        # 收集所有子地形的高度范围
        for row_data in self.terrain_data:
            for terrain_output in row_data:
                for geom in terrain_output.geometries:
                    if geom.height_map is not None:
                        min_heights.append(geom.height_map.min_height)
                        max_heights.append(geom.height_map.max_height)

        # 收集边界的高度范围
        if self.border_data is not None:
            for geom in self.border_data:
                if geom.height_map is not None:
                    min_heights.append(geom.height_map.min_height)
                    max_heights.append(geom.height_map.max_height)

        # 计算全局高度范围
        if min_heights and max_heights:
            self.global_height_range = (
                torch.tensor(min_heights, device=self.device).min().item(),
                torch.tensor(max_heights, device=self.device).max().item()
            )
        else:
            self.global_height_range = (0.0, 0.0)  # 无数据时默认范围

    def _generate_uniform(self, low: float, high: float) -> torch.Tensor:
        """
        兼容旧PyTorch版本的均匀分布随机数生成函数
        替代 torch.empty(..., generator=...).uniform_(low, high)
        Args:
            low: 最小值
            high: 最大值
        Returns:
            torch.Tensor: 形状为(1,)的随机数张量
        """
        return (high - low) * torch.rand(1, generator=self.torch_rng, device=self.device) + low

    def get_terrain_at(self, row: int, col: int) -> Optional[TerrainOutput]:
        """
        获取指定网格位置的地形数据
        Args:
            row: 行索引
            col: 列索引
        Returns:
            Optional[TerrainOutput]: 地形数据（索引越界返回None）
        """
        if 0 <= row < self.cfg.num_rows and 0 <= col < self.cfg.num_cols:
            return self.terrain_data[row][col]
        else:
            print(f"警告：索引 ({row}, {col}) 超出地形网格范围 ({self.cfg.num_rows}, {self.cfg.num_cols})")
            return None

    def get_all_terrain_data(self) -> Tuple[List[List[TerrainOutput]], Optional[List[TerrainGeometry]]]:
        """
        获取所有地形数据和边界数据
        Returns:
            Tuple: (地形网格数据, 边界数据)
        """
        return self.terrain_data, self.border_data

    def get_global_height_range(self) -> Tuple[float, float]:
        """
        获取全局地形的高度范围
        Returns:
            Tuple[float, float]: (最小高度, 最大高度)
        """
        return self.global_height_range

    def get_terrain_origins(self) -> torch.Tensor:
        """
        获取所有子地形的spawn原点（机器人生成位置）
        Returns:
            torch.Tensor: 形状为(num_rows, num_cols, 3)的张量
        """
        return self.terrain_origins.clone()  # 返回副本，防止外部修改


# ------------------------------ 内置子地形实现（兼容配置） ------------------------------
@dataclass
class FlatTerrainCfg(SubTerrainCfg):
    """
    平坦地形配置类
    难度不影响地形，高度固定
    """
    height: float = 0.0  # 平坦地形的固定高度

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成平坦地形数据
        Args:
            difficulty: 难度系数（无实际作用）
            rng: 随机数生成器（无实际作用）
            device: 数据存储设备
        Returns:
            TerrainOutput: 平坦地形数据
        """
        # 生成平坦高度图（所有高度值=固定高度）
        height_data = torch.full(
            self.resolution, self.height, dtype=torch.float32, device=device
        )

        # 子地形本地坐标：spawn原点在地形中心，z轴为地形高度
        local_origin = torch.tensor(
            [self.size[0] / 2, self.size[1] / 2, self.height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标：左下角在 (0,0,0)
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=(0.3, 0.7, 0.3, 1.0)  # 绿色（平坦地形标识色）
            )]
        )


@dataclass
class RoughTerrainCfg(SubTerrainCfg):
    """
    粗糙地形配置类
    难度越高，地形起伏越大，表面越不规则
    """
    base_height: float = 0.0  # 地形基础高度
    max_height_diff: float = 0.8  # 最大高度差（难度=1时达到）
    noise_scale: float = 5.0  # 噪声平滑程度（值越大越平滑）

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成粗糙地形数据
        核心流程：生成随机噪声 → 高斯平滑 → 归一化 → 难度缩放
        Args:
            difficulty: 难度系数 [0,1]
            rng: 随机数生成器
            device: 数据存储设备
        Returns:
            TerrainOutput: 粗糙地形数据
        """
        # 生成高斯随机噪声
        noise = torch.randn(
            self.resolution, generator=rng,
            device=device, dtype=torch.float32
        )

        # 高斯平滑噪声（使地形过渡自然）
        smoothed_noise = gaussian_filter_torch(noise, sigma=self.noise_scale)

        # 归一化噪声到 [-1, 1] 范围（避免除零）
        noise_min = smoothed_noise.min()
        noise_max = smoothed_noise.max()
        smoothed_noise = (smoothed_noise - noise_min) / (noise_max - noise_min + 1e-6)
        smoothed_noise = smoothed_noise * 2 - 1

        # 根据难度控制高度差
        height_diff = self.max_height_diff * difficulty
        height_data = self.base_height + smoothed_noise * height_diff / 2

        # 计算地形平均高度（作为spawn原点的z坐标）
        avg_height = height_data.mean().item()
        local_origin = torch.tensor(
            [self.size[0] / 2, self.size[1] / 2, avg_height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        # 基于高度生成颜色（低=绿，高=黄）
        height_normalized = (height_data - height_data.min()) / (height_data.max() - height_data.min() + 1e-6)
        mean_norm = height_normalized.mean().item()
        color = (
            0.2 + mean_norm * 0.6,  # R: 0.2~0.8（高度越高越红）
            0.6 - mean_norm * 0.3,  # G: 0.6~0.3（高度越高越绿）
            0.1,                    # B: 固定0.1
            1.0                     # A: 不透明
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=color
            )]
        )


@dataclass
class StepTerrainCfg(SubTerrainCfg):
    """
    台阶地形配置类
    难度越高，台阶数量越多、高度差越大
    """
    min_step_height: float = 0.1  # 最小台阶高度
    max_step_height: float = 0.5  # 最大台阶高度
    min_step_count: int = 2       # 最小台阶数量
    max_step_count: int = 8       # 最大台阶数量

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成台阶地形数据
        核心流程：计算台阶数量和高度 → 生成台阶高度图 → 平滑边缘
        Args:
            difficulty: 难度系数 [0,1]
            rng: 随机数生成器
            device: 数据存储设备
        Returns:
            TerrainOutput: 台阶地形数据
        """
        # 根据难度计算台阶数量和高度
        step_count = int(self.min_step_count + (self.max_step_count - self.min_step_count) * difficulty)
        step_height = self.min_step_height + (self.max_step_height - self.min_step_height) * difficulty

        # 初始化高度图
        height_data = torch.zeros(self.resolution, dtype=torch.float32, device=device)
        # 计算每个台阶对应的像素数
        resolution_per_step = self.resolution[1] // step_count

        current_height = 0.0  # 当前台阶高度
        for i in range(step_count):
            # 随机决定台阶上升/下降（增加地形多样性）
            if i > 0 and torch.rand(1, generator=rng, device=device).item() > 0.5:
                current_height += step_height * torch.rand(1, generator=rng, device=device).uniform_(0.7, 1.0).item()
            else:
                current_height = max(0.0, current_height - step_height * torch.rand(1, generator=rng,
                                                                                    device=device).uniform_(0.5,
                                                                                                            1.0).item())

            # 填充当前台阶的高度值
            start_col = i * resolution_per_step
            end_col = (i + 1) * resolution_per_step if i < step_count - 1 else self.resolution[1]
            height_data[:, start_col:end_col] = current_height

        # 平滑台阶边缘（减少锯齿）
        height_data = gaussian_filter_torch(height_data, sigma=1.0)

        # 计算平均高度作为spawn原点z坐标
        avg_height = height_data.mean().item()
        local_origin = torch.tensor(
            [self.size[0] / 2, self.size[1] / 2, avg_height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        # 基于高度生成颜色（低=蓝，高=红）
        height_normalized = (height_data - height_data.min()) / (height_data.max() - height_data.min() + 1e-6)
        mean_norm = height_normalized.mean().item()
        color = (
            mean_norm * 0.7 + 0.2,  # R: 0.2~0.9
            0.3 - mean_norm * 0.2,  # G: 0.3~0.1
            0.7 - mean_norm * 0.6,  # B: 0.7~0.1
            1.0
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=color
            )]
        )


@dataclass
class SlopeTerrainCfg(SubTerrainCfg):
    """
    斜坡地形配置类
    难度越高，坡度越大
    """
    min_slope: float = 0.05  # 最小坡度（rise/run，高度/长度）
    max_slope: float = 0.3   # 最大坡度
    slope_direction: Literal["x", "y"] = "x"  # 斜坡方向（x或y轴）

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成斜坡地形数据
        核心流程：计算坡度 → 生成斜坡高度图 → 添加噪声 → 着色
        Args:
            difficulty: 难度系数 [0,1]
            rng: 随机数生成器
            device: 数据存储设备
        Returns:
            TerrainOutput: 斜坡地形数据
        """
        # 根据难度计算当前坡度
        slope = self.min_slope + (self.max_slope - self.min_slope) * difficulty

        # 解析分辨率和尺寸
        resolution_y, resolution_x = self.resolution
        size_x, size_y = self.size

        # 生成斜坡高度图
        if self.slope_direction == "x":
            # 沿x轴方向倾斜
            x_coords = torch.linspace(0, size_x, resolution_x, device=device, dtype=torch.float32)
            height_data = slope * x_coords.unsqueeze(0).repeat(resolution_y, 1)
        else:
            # 沿y轴方向倾斜
            y_coords = torch.linspace(0, size_y, resolution_y, device=device, dtype=torch.float32)
            height_data = slope * y_coords.unsqueeze(1).repeat(1, resolution_x)

        # 添加少量噪声使斜坡更真实（避免完全光滑）
        noise = 0.02 * torch.randn(
            self.resolution, generator=rng,
            device=device, dtype=torch.float32
        )
        height_data += noise

        # 计算平均高度
        avg_height = height_data.mean().item()
        local_origin = torch.tensor(
            [self.size[0] / 2, self.size[1] / 2, avg_height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        # 基于坡度生成颜色（坡度越大越红）
        slope_norm = (slope - self.min_slope) / (self.max_slope - self.min_slope + 1e-6)
        color = (
            0.3 + slope_norm * 0.6,  # R: 0.3~0.9
            0.7 - slope_norm * 0.5,  # G: 0.7~0.2
            0.2,                    # B: 固定0.2
            1.0
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=color
            )]
        )


# ------------------------------ 兼容用户配置的地形类型 ------------------------------
@dataclass
class BoxFlatTerrainCfg(FlatTerrainCfg):
    """
    兼容用户配置的平坦地形（Box前缀）
    仅为兼容命名，无额外逻辑
    """
    pass


@dataclass
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
    """
    金字塔台阶地形配置类（兼容用户配置）
    地形特征：中心平台 → 向外逐级升高的台阶 → 边界
    """
    border_width: float = 1.0  # 边界宽度
    step_height_range: Tuple[float, float] = (0.0, 0.1)  # 台阶高度范围
    step_width: float = 0.3  # 台阶宽度
    platform_width: float = 3.0  # 中心平台宽度
    holes: bool = False  # 是否添加孔洞

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成金字塔台阶地形数据
        核心流程：计算中心平台 → 生成台阶高度 → 添加边界 → 可选孔洞
        Args:
            difficulty: 难度系数 [0,1]
            rng: 随机数生成器
            device: 数据存储设备
        Returns:
            TerrainOutput: 金字塔台阶地形数据
        """
        # 根据难度计算台阶高度
        min_step_h, max_step_h = self.step_height_range
        step_height = min_step_h + (max_step_h - min_step_h) * difficulty

        # 初始化高度图
        height_data = torch.zeros(self.resolution, dtype=torch.float32, device=device)
        terrain_width, terrain_depth = self.size
        res_y, res_x = self.resolution

        # 计算中心平台范围
        platform_half = self.platform_width / 2
        center_x = terrain_width / 2
        center_y = terrain_depth / 2

        # 生成坐标网格（用于计算每个点的高度）
        x_coords = torch.linspace(0, terrain_width, res_x, device=device)
        y_coords = torch.linspace(0, terrain_depth, res_y, device=device)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # 计算每个点到中心平台的距离（x和y方向）
        dist_x = torch.abs(xx - center_x) - platform_half
        dist_y = torch.abs(yy - center_y) - platform_half
        dist = torch.maximum(dist_x, dist_y)  # 取x/y方向的最大距离
        dist = torch.maximum(dist, torch.tensor(0.0, device=device))  # 限制为非负

        # 计算每个点的台阶数量和高度
        num_steps = torch.ceil(dist / self.step_width).int()
        height_data = num_steps.float() * step_height

        # 添加边界（高度稍高）
        border_half = self.border_width / 2
        border_mask = (xx < border_half) | (xx > terrain_width - border_half) | \
                      (yy < border_half) | (yy > terrain_depth - border_half)
        height_data[border_mask] = step_height * 2

        # 可选：添加孔洞（增加地形难度）
        if self.holes:
            num_holes = int(5 * difficulty) + 1  # 难度越高孔洞越多
            for _ in range(num_holes):
                # 随机生成孔洞位置和半径
                hole_x = torch.rand(1, generator=rng, device=device).item() * terrain_width
                hole_y = torch.rand(1, generator=rng, device=device).item() * terrain_depth
                hole_radius = torch.rand(1, generator=rng, device=device).item() * 0.5 + 0.3
                # 计算孔洞区域
                hole_mask = (xx - hole_x) ** 2 + (yy - hole_y) ** 2 < hole_radius ** 2
                height_data[hole_mask] = -0.1  # 孔洞向下凹陷

        # 计算平均高度
        avg_height = height_data.mean().item()
        local_origin = torch.tensor(
            [center_x, center_y, avg_height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        # 基于高度生成颜色
        height_normalized = (height_data - height_data.min()) / (height_data.max() - height_data.min() + 1e-6)
        color = (
            0.4 + height_normalized.mean().item() * 0.5,
            0.5 - height_normalized.mean().item() * 0.3,
            0.3,
            1.0
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=color
            )]
        )


@dataclass
class BoxInvertedPyramidStairsTerrainCfg(SubTerrainCfg):
    """
    倒金字塔台阶地形配置类（兼容用户配置）
    地形特征：边界 → 向内逐级降低的台阶 → 中心平台
    """
    border_width: float = 1.0  # 边界宽度
    step_height_range: Tuple[float, float] = (0.0, 0.1)  # 台阶高度范围
    step_width: float = 0.3  # 台阶宽度
    platform_width: float = 3.0  # 中心平台宽度
    holes: bool = False  # 是否添加孔洞

    def function(self, difficulty: float, rng: torch.Generator, device: str) -> TerrainOutput:
        """
        生成倒金字塔台阶地形数据
        核心流程：计算边界 → 生成向内降低的台阶 → 中心平台 → 可选孔洞
        Args:
            difficulty: 难度系数 [0,1]
            rng: 随机数生成器
            device: 数据存储设备
        Returns:
            TerrainOutput: 倒金字塔台阶地形数据
        """
        # 根据难度计算台阶高度
        min_step_h, max_step_h = self.step_height_range
        step_height = min_step_h + (max_step_h - min_step_h) * difficulty

        # 初始化高度图
        height_data = torch.zeros(self.resolution, dtype=torch.float32, device=device)
        terrain_width, terrain_depth = self.size
        res_y, res_x = self.resolution

        # 计算边界范围
        border_half = self.border_width / 2
        inner_width = terrain_width - 2 * border_half
        inner_depth = terrain_depth - 2 * border_half

        # 生成坐标网格
        x_coords = torch.linspace(0, terrain_width, res_x, device=device)
        y_coords = torch.linspace(0, terrain_depth, res_y, device=device)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # 计算每个点到边界的距离
        dist_x = torch.minimum(xx - border_half, inner_width - (xx - border_half))
        dist_y = torch.minimum(yy - border_half, inner_depth - (yy - border_half))
        dist = torch.minimum(dist_x, dist_y)  # 取x/y方向的最小距离
        dist = torch.maximum(dist, torch.tensor(0.0, device=device))

        # 计算台阶高度（越靠近中心越低）
        max_steps = inner_width / (2 * self.step_width)
        num_steps = torch.floor((max_steps - dist / self.step_width)).int()
        height_data = num_steps.float() * step_height

        # 中心平台（高度为0）
        platform_half = self.platform_width / 2
        center_x = terrain_width / 2
        center_y = terrain_depth / 2
        platform_mask = (torch.abs(xx - center_x) < platform_half) & \
                        (torch.abs(yy - center_y) < platform_half)
        height_data[platform_mask] = 0.0

        # 边界高度（最高）
        border_mask = (xx < border_half) | (xx > terrain_width - border_half) | \
                      (yy < border_half) | (yy > terrain_depth - border_half)
        height_data[border_mask] = step_height * torch.ceil(max_steps).float()

        # 可选：添加孔洞
        if self.holes:
            num_holes = int(5 * difficulty) + 1
            for _ in range(num_holes):
                hole_x = torch.rand(1, generator=rng, device=device).item() * terrain_width
                hole_y = torch.rand(1, generator=rng, device=device).item() * terrain_depth
                hole_radius = torch.rand(1, generator=rng, device=device).item() * 0.5 + 0.3
                hole_mask = (xx - hole_x) ** 2 + (yy - hole_y) ** 2 < hole_radius ** 2
                height_data[hole_mask] = -0.1  # 孔洞向下凹陷

        # 计算平均高度
        avg_height = height_data.mean().item()
        local_origin = torch.tensor(
            [center_x, center_y, avg_height],
            device=device, dtype=torch.float32
        )

        # 高度图本地坐标
        height_map_origin = torch.tensor(
            [0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        # 基于高度生成颜色
        height_normalized = (height_data - height_data.min()) / (height_data.max() - height_data.min() + 1e-6)
        color = (
            0.4 + height_normalized.mean().item() * 0.5,
            0.3 - height_normalized.mean().item() * 0.2,
            0.5,
            1.0
        )

        return TerrainOutput(
            origin=local_origin,
            geometries=[TerrainGeometry(
                height_map=HeightMapData(
                    data=height_data,
                    origin=height_map_origin,
                    size=self.size,
                    resolution=self.resolution
                ),
                color=color
            )]
        )


# ------------------------------ 完整使用示例 ------------------------------
if __name__ == "__main__":
    """
    地形生成器使用示例
    演示配置创建、生成器初始化、地形生成和数据访问的完整流程
    """
    # 自动选择设备（优先CUDA）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # 1. 配置子地形（定义可用的地形类型和参数）
    sub_terrains = {
        "flat": BoxFlatTerrainCfg(
            proportion=0.4,  # 40% 概率选中
            resolution=(128, 128),
            height=0.0
        ),
        "pyramid_stairs": BoxPyramidStairsTerrainCfg(
            proportion=0.3,  # 30% 概率选中
            resolution=(128, 128),
            border_width=1.0,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            holes=False
        ),
        "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.3,  # 30% 概率选中
            resolution=(128, 128),
            border_width=1.0,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            holes=False
        )
    }

    # 2. 配置生成器全局参数
    generator_cfg = TerrainGeneratorCfg(
        seed=None,                # 随机种子（None=自动生成）
        curriculum=True,          # 启用课程式难度
        size=(8.0, 8.0),          # 每个子地形8x8米
        border_width=20.0,        # 边界宽度20米
        border_height=1.0,        # 边界高度1米
        num_rows=10,              # 10行地形
        num_cols=20,              # 20列地形
        color_scheme="height",    # 按高度着色
        sub_terrains=sub_terrains,# 子地形配置
        difficulty_range=(0.0, 1.0),  # 难度范围
        add_border=True           # 生成边界
    )

    # 3. 创建生成器实例并生成地形
    generator = TerrainGenerator(cfg=generator_cfg, device=device)
    generator.generate()

    # 4. 访问生成的地形数据（示例）
    print("\n=== 生成数据访问示例 ===")
    # 获取(0,0)位置的地形数据（课程式难度中最简单的地形）
    terrain_0_0 = generator.get_terrain_at(0, 0)
    if terrain_0_0:
        print(f"\n地形 (0,0) 信息：")
        print(f"  spawn原点：{terrain_0_0.origin.cpu().numpy()}")
        print(f"  高度范围：[{terrain_0_0.geometries[0].height_map.min_height:.3f}, "
              f"{terrain_0_0.geometries[0].height_map.max_height:.3f}]")
        print(f"  高度图形状：{terrain_0_0.geometries[0].height_map.data.shape}")
        print(f"  颜色：{terrain_0_0.geometries[0].color}")

    # 5. 保存高度图示例（用于离线分析或可视化）
    if terrain_0_0:
        height_map_data = terrain_0_0.geometries[0].height_map.data
        torch.save(height_map_data, "terrain_0_0_height_map.pt")
        print(f"\n地形 (0,0) 高度图已保存为：terrain_0_0_height_map.pt")

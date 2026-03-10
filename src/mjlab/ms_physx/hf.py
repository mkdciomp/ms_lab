
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable


# -----------------------------------------------------------------------------
# Configuration Classes (replaced Isaac Lab's hf_terrains_cfg)
# 配置类：替换Isaac Lab的hf_terrains_cfg配置系统
# 每个地形类型对应一个配置类，继承自基础配置类，包含该地形的所有参数
# -----------------------------------------------------------------------------
@dataclass
class HfBaseTerrainCfg:
    """Base configuration for height field terrains.
    高度场地形的基础配置类
    所有地形配置类的父类，定义通用参数
    """
    # 地形物理尺寸（宽度，长度），单位：米
    size: Tuple[float, float] = (8.0, 8.0)  # (width, length) in meters
    # 水平/垂直比例（米/像素），控制高度场的分辨率
    proportion = 0.05
    # 坡度阈值（可选），用于限制地形最大坡度
    slope_threshold = None
    # 边界宽度，用于避免边缘伪影
    border_width: float = 0.2  # 边界宽度，用于避免边缘 artifacts
    # 计算设备（CPU/GPU）
    device = "cpu"
    # 数据类型
    dtype = torch.float32

    # 水平比例属性（兼容旧接口）
    @property
    def horizontal_scale(self):
        return self.proportion

    # 垂直比例属性（兼容旧接口）
    @property
    def vertical_scale(self):
        return self.proportion

    # 地形生成函数（需在子类中实现）
    def func(self, difficulty: float) -> torch.Tensor:
        """Generate terrain using this configuration.
        使用当前配置生成地形高度场
        Args:
            difficulty: 地形难度系数（0-1），控制地形复杂度/强度
        Returns:
            地形高度场张量（2D）
        """
        raise NotImplementedError("Subclasses must implement the func() method")


@dataclass
class HfFlatTerrainCfg(HfBaseTerrainCfg):
    """Configuration for flat terrain.
    平坦地形配置类
    """

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate flat terrain using this configuration.
        生成平坦地形高度场
        """
        return flat_terrain(difficulty, self)


@dataclass
class HfRandomUniformTerrainCfg(HfBaseTerrainCfg):
    """Configuration for random uniform terrain.
    随机均匀地形配置类
    地形高度从指定范围内均匀采样
    """
    # 噪声高度范围（最小/最大高度），单位：米
    noise_range: Tuple[float, float] = (-0.5, 0.5)  # min/max height in meters
    # 高度步长，控制高度的离散精度
    noise_step: float = 0.05  # step between possible heights in meters
    # 下采样比例，用于生成低分辨率初始网格后上采样，减少计算量
    downsampled_scale: Optional[float] = None  # downsampling scale for initial grid

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate random uniform terrain using this configuration."""
        return random_uniform_terrain(difficulty, self)


@dataclass
class HfPyramidSlopedTerrainCfg(HfBaseTerrainCfg):
    """Configuration for pyramid sloped terrain.
    金字塔斜坡地形配置类
    中心为平坦平台，向四周逐渐倾斜的金字塔形地形
    """
    # 坡度范围（最小/最大坡度），坡度=高度变化/水平距离
    slope_range: Tuple[float, float] = (0.1, 0.5)  # min/max slope (height/width)
    # 中心平坦平台宽度，单位：米
    platform_width: float = 2.0  # width of the flat platform in meters
    # 是否反转（平台在底部）
    inverted: bool = False  # whether to invert the pyramid (platform at bottom)

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate pyramid sloped terrain using this configuration."""
        return pyramid_sloped_terrain(difficulty, self)


@dataclass
class HfPyramidStairsTerrainCfg(HfBaseTerrainCfg):
    """Configuration for pyramid stairs terrain.
    金字塔阶梯地形配置类
    中心为平坦平台，向四周呈阶梯状上升/下降的金字塔形地形
    """
    # 台阶高度范围（最小/最大高度），单位：米
    step_height_range: Tuple[float, float] = (0.1, 0.3)  # min/max step height in meters
    # 每个台阶的宽度，单位：米
    step_width: float = 0.5  # width of each step in meters
    # 中心平坦平台宽度，单位：米
    platform_width: float = 2.0  # width of the flat platform in meters
    # 是否反转（平台在底部）
    inverted: bool = False  # whether to invert the stairs (platform at bottom)

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate pyramid stairs terrain using this configuration."""
        return pyramid_stairs_terrain(difficulty, self)


@dataclass
class HfDiscreteObstaclesTerrainCfg(HfBaseTerrainCfg):
    """Configuration for discrete obstacles terrain.
    离散障碍物地形配置类
    中心为平坦平台，周围随机分布柱状障碍物（可正可负高度）
    """
    # 障碍物数量
    num_obstacles: int = 20  # number of obstacles
    # 障碍物高度范围（最小/最大高度），单位：米
    obstacle_height_range: Tuple[float, float] = (0.2, 0.8)  # min/max obstacle height in meters
    # 障碍物宽度范围（最小/最大宽度），单位：米
    obstacle_width_range: Tuple[float, float] = (0.4, 1.0)  # min/max obstacle width in meters
    # 障碍物高度模式："choice"（从预设值选择）或 "fixed"（固定值）
    obstacle_height_mode: str = "choice"  # "choice" or "fixed"
    # 中心平坦平台宽度，单位：米
    platform_width: float = 2.0  # width of the flat platform in meters

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate discrete obstacles terrain using this configuration."""
        return discrete_obstacles_terrain(difficulty, self)


@dataclass
class HfWaveTerrainCfg(HfBaseTerrainCfg):
    """Configuration for wave terrain.
    波浪地形配置类
    整个地形覆盖正弦/余弦波浪图案
    """
    # 波幅范围（最小/最大振幅），单位：米
    amplitude_range: Tuple[float, float] = (0.1, 0.5)  # min/max wave amplitude in meters
    # 沿长度方向的波数
    num_waves: int = 4  # number of waves along the length

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate wave terrain using this configuration."""
        return wave_terrain(difficulty, self)


@dataclass
class HfSteppingStonesTerrainCfg(HfBaseTerrainCfg):
    """Configuration for stepping stones terrain.
    垫脚石地形配置类
    地形由一系列离散的"石头"组成，石头之间是凹陷的孔洞
    """
    # 石头宽度范围（最小/最大宽度），单位：米
    stone_width_range: Tuple[float, float] = (0.3, 0.8)  # min/max stone width in meters
    # 石头间距范围（最小/最大距离），单位：米
    stone_distance_range: Tuple[float, float] = (0.1, 0.5)  # min/max distance between stones in meters
    # 石头最大高度（可负），单位：米
    stone_height_max: float = 0.3  # max stone height (can be negative) in meters
    # 石头之间孔洞的深度，单位：米
    holes_depth: float = -0.5  # depth of the holes between stones in meters
    # 中心平坦平台宽度，单位：米
    platform_width: float = 2.0  # width of the flat platform in meters

    def func(self, difficulty: float) -> torch.Tensor:
        """Generate stepping stones terrain using this configuration."""
        return stepping_stones_terrain(difficulty, self)


# -----------------------------------------------------------------------------
# Decorator (replaced Isaac Lab's height_field_to_mesh)
# 装饰器：兼容Isaac Lab的height_field_to_mesh接口
# 原装饰器将高度场转换为网格，这里保留接口但直接返回PyTorch张量
# -----------------------------------------------------------------------------
def height_field_to_mesh(func):
    """Decorator to keep the same interface as Isaac Lab.

    The original decorator converted height fields to meshes, but we keep it
    for compatibility while returning the PyTorch tensor directly.

    装饰器说明：
    - 保持与Isaac Lab相同的接口
    - 原装饰器将高度场转换为网格数据结构
    - 此处简化实现，直接返回PyTorch张量以保证兼容性
    """

    def wrapper(difficulty: float, cfg: HfBaseTerrainCfg) -> torch.Tensor:
        return func(difficulty, cfg)

    return wrapper


# -----------------------------------------------------------------------------
# Terrain Generation Functions
# 地形生成函数：每个函数对应一种地形类型的具体生成逻辑
# -----------------------------------------------------------------------------
@height_field_to_mesh
def flat_terrain(difficulty: float, cfg: HfFlatTerrainCfg) -> torch.Tensor:
    """Generate a completely flat terrain with fixed base height.

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain (not used here).
        cfg: The configuration for the flat terrain.

    Returns:
        The height field of the flat terrain as a 2D torch tensor.
        All values are set to the base height (discretized).

    生成平坦地形：
    - 所有高度值都为0（基础高度）
    - difficulty参数无实际作用
    - 返回离散化的2D高度场张量
    """
    # 转换为离散像素单位：物理尺寸 / 像素比例 = 像素数量
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # 基础高度离散化：0米高度转换为像素单位
    base_height_discrete = int(0.0 / cfg.vertical_scale)

    # 创建全为基础高度的张量（int16类型节省内存）
    flat_hf = torch.full(
        (width_pixels, length_pixels),
        base_height_discrete,
        dtype=torch.int16
    )

    return flat_hf


@height_field_to_mesh
def random_uniform_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg) -> torch.Tensor:
    """Generate a terrain with height sampled uniformly from a specified range.

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.

    生成随机均匀地形：
    1. 在低分辨率网格上随机采样高度值
    2. 使用双线性插值上采样到目标分辨率
    3. 离散化高度值到指定步长
    """
    # 参数检查：下采样比例不能小于基础比例
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    # 高度范围离散化
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # 创建可选高度值范围
    height_range = torch.arange(height_min, height_max + height_step, height_step)

    # 在低分辨率网格上随机采样高度
    height_field_downsampled = torch.randint(
        0, len(height_range), (width_downsampled, length_downsampled)
    )
    height_field_downsampled = height_range[height_field_downsampled]

    # 创建插值网格坐标
    x = torch.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = torch.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)

    # 创建上采样目标网格坐标
    x_upsampled = torch.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = torch.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)

    # 生成网格坐标用于插值 (H, W, 2)
    grid_x, grid_y = torch.meshgrid(x_upsampled, y_upsampled, indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)  # grid_sample要求的格式

    # 调整输入形状以适应grid_sample: (batch, channel, height, width)
    hf_input = height_field_downsampled.unsqueeze(0).unsqueeze(0).float()

    # 双线性插值上采样
    z_upsampled = F.grid_sample(
        hf_input, grid.unsqueeze(0).float(),
        mode='bilinear', padding_mode='border', align_corners=True
    )
    z_upsampled = z_upsampled.squeeze(0).squeeze(0)

    # 将插值后的高度值四舍五入到最近的高度步长
    return z_upsampled.int()


@height_field_to_mesh
def pyramid_sloped_terrain(difficulty: float, cfg: HfPyramidSlopedTerrainCfg) -> torch.Tensor:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    生成金字塔斜坡地形：
    1. 根据难度系数计算当前坡度（0-1映射到坡度范围）
    2. 创建中心对称的斜坡高度场
    3. 裁剪出中心平坦平台
    4. 离散化高度值
    """
    # 根据难度系数和反转标志计算当前坡度
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # 计算最大高度：坡度 * 半宽 / 垂直比例
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # 计算地形中心像素坐标
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # 创建地形网格坐标
    x = torch.arange(0, width_pixels)
    y = torch.arange(0, length_pixels)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # 将坐标归一化到中心（0-1范围）
    xx_normalized = (center_x - torch.abs(center_x - xx)) / center_x
    yy_normalized = (center_y - torch.abs(center_y - yy)) / center_y

    # 创建金字塔斜坡表面：高度 = 最大高度 * x归一化值 * y归一化值
    hf_raw = height_max * xx_normalized * yy_normalized

    # 创建中心平坦平台
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]  # 平台高度
    # 裁剪高度场，使中心区域保持平台高度
    hf_raw = torch.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # 离散化高度值
    return hf_raw.int()


@height_field_to_mesh
def pyramid_stairs_terrain(difficulty: float, cfg: HfPyramidStairsTerrainCfg) -> torch.Tensor:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    生成金字塔阶梯地形：
    1. 根据难度系数计算台阶高度
    2. 从外到内逐层创建阶梯
    3. 直到剩余区域小于平台宽度
    """
    # 根据难度系数计算当前台阶高度
    step_height = (cfg.step_height_range[0] + difficulty *
                   (cfg.step_height_range[1] - cfg.step_height_range[0]))

    # 反转标志：台阶高度取反
    if cfg.inverted:
        step_height *= -1

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = int(cfg.step_width / cfg.horizontal_scale + 1)
    scale_1 = -1 if cfg.inverted else 1

    # 离散化台阶高度
    step_height = int(step_height / cfg.vertical_scale + scale_1)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale + 1)

    # 创建初始平坦高度场
    hf_raw = torch.zeros((width_pixels, length_pixels), dtype=torch.float32)

    # 逐层添加阶梯
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels

    # 当剩余区域大于平台宽度时继续创建阶梯
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        # 向内收缩台阶边界
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width

        # 增加台阶高度
        current_step_height += step_height

        # 设置当前台阶的高度值
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    # 离散化高度值
    return hf_raw.int()


@height_field_to_mesh
def discrete_obstacles_terrain(difficulty: float, cfg: HfDiscreteObstaclesTerrainCfg) -> torch.Tensor:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    生成离散障碍物地形：
    1. 根据难度系数计算障碍物高度
    2. 随机生成指定数量的障碍物（位置、大小、高度）
    3. 保证中心平台区域无障碍物
    4. 离散化高度值
    """
    # 根据难度系数计算障碍物高度
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
            cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale + 1)
    obs_height = int(obs_height / cfg.vertical_scale + 1)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale + 1)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale + 1)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale + 1)

    # 创建障碍物尺寸的离散范围（步长4以减少计算量）
    obs_width_range = torch.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = torch.arange(obs_width_min, obs_width_max, 4)
    obs_x_range = torch.arange(0, width_pixels, 4)
    obs_y_range = torch.arange(0, length_pixels, 4)

    # 创建初始平坦高度场
    hf_raw = torch.zeros((width_pixels, length_pixels), dtype=torch.float32)

    # 生成指定数量的障碍物
    for _ in range(cfg.num_obstacles):
        # 采样障碍物高度
        if cfg.obstacle_height_mode == "choice":
            # 从预设值中选择：-全高、-半高、+半高、+全高
            height_options = torch.tensor([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
            height = height_options[torch.randint(0, len(height_options), (1,))].item()
        elif cfg.obstacle_height_mode == "fixed":
            # 使用固定高度
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")

        # 采样障碍物尺寸
        width = obs_width_range[torch.randint(0, len(obs_width_range), (1,))].item()
        length = obs_length_range[torch.randint(0, len(obs_length_range), (1,))].item()

        # 采样障碍物位置
        x_start = obs_x_range[torch.randint(0, len(obs_x_range), (1,))].item()
        y_start = obs_y_range[torch.randint(0, len(obs_y_range), (1,))].item()

        # 确保障碍物不超出地形边界
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length

        # 设置障碍物高度
        hf_raw[int(x_start):int(x_start + width), int(y_start):int(y_start + length)] = height

    # 保证中心平台区域无障碍物（设为0）
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    # 离散化高度值
    return hf_raw.int()


@height_field_to_mesh
def wave_terrain(difficulty: float, cfg: HfWaveTerrainCfg) -> torch.Tensor:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.

    生成波浪地形：
    1. 根据难度系数计算波幅
    2. 计算波数和波长
    3. 使用正弦/余弦函数生成波浪图案
    4. 离散化高度值
    """
    # 检查波数必须为正整数
    if cfg.num_waves <= 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # 根据难度系数计算波幅
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # 计算波数：nu = 2π / λ（λ为波长）
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * torch.pi / wave_length

    # 创建地形网格坐标
    x = torch.arange(0, width_pixels)
    y = torch.arange(0, length_pixels)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # 创建初始平坦高度场
    hf_raw = torch.zeros((width_pixels, length_pixels), dtype=torch.float32)

    # 添加波浪图案：正弦(x方向) + 余弦(y方向)
    hf_raw += amplitude_pixels * (torch.cos(yy * wave_number) + torch.sin(xx * wave_number))

    # 离散化高度值
    return hf_raw.int()


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: HfSteppingStonesTerrainCfg) -> torch.Tensor:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D torch tensor with discretized heights.
        The shape of the tensor is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    生成垫脚石地形：
    1. 根据难度系数调整石头尺寸和间距（难度越高，石头越小、间距越大）
    2. 初始化地形为孔洞深度
    3. 按行列填充石头（随机高度）
    4. 保证中心平台区域平坦
    5. 离散化高度值
    """
    # 根据难度系数计算石头尺寸和间距：
    # - 难度越高，石头越小（从最大值向最小值变化）
    # - 难度越高，间距越大（从最小值向最大值变化）
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
            cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # 转换为离散像素单位
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_distance = int(stone_distance / cfg.horizontal_scale + 1)
    stone_width = int(stone_width / cfg.horizontal_scale + 1)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale + 1)

    # 创建石头高度范围（-max_height 到 max_height）
    stone_height_range = torch.arange(-stone_height_max - 1, stone_height_max, step=1)

    # 初始化地形为孔洞深度
    hf_raw = torch.full((width_pixels, length_pixels), holes_depth, dtype=torch.float32)

    # 添加石头
    start_x, start_y = 0, 0

    # 如果地形长度 >= 宽度，按列填充
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # 确保石头不超出y轴边界
            stop_y = min(length_pixels, start_y + stone_width)

            # 随机采样x起始位置
            start_x = torch.randint(0, stone_width, (1,)).item()
            stop_x = max(0, start_x - stone_distance)

            # 填充第一个石头
            if stop_x > 0:
                hf_raw[0:stop_x, start_y:stop_y] = stone_height_range[
                    torch.randint(0, len(stone_height_range), (1,))].item()

            # 按行填充石头
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = stone_height_range[
                    torch.randint(0, len(stone_height_range), (1,))].item()
                start_x += stone_width + stone_distance

            # 更新y位置
            start_y += stone_width + stone_distance
    else:
        # 否则按行填充
        while start_x < width_pixels:
            # 确保石头不超出x轴边界
            stop_x = min(width_pixels, start_x + stone_width)

            # 随机采样y起始位置
            start_y = torch.randint(0, stone_width, (1,)).item()
            stop_y = max(0, start_y - stone_distance)

            # 填充第一个石头
            if stop_y > 0:
                hf_raw[start_x:stop_x, 0:stop_y] = stone_height_range[
                    torch.randint(0, len(stone_height_range), (1,))].item()

            # 按列填充石头
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = stone_height_range[
                    torch.randint(0, len(stone_height_range), (1,))].item()
                start_y += stone_width + stone_distance

            # 更新x位置
            start_x += stone_width + stone_distance

    # 添加中心平坦平台
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    # 离散化高度值
    return hf_raw.int()


# -----------------------------------------------------------------------------
# Example Usage
# 示例用法：展示如何使用各个配置类生成不同类型的地形
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

    # 设置难度（0-1之间）
    difficulty = 0.5

    # 1. 随机均匀地形 - 直接通过配置类调用func()方法
    random_cfg = HfRandomUniformTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,  # 替代horizontal_scale/vertical_scale
        noise_range=(-0.5, 0.5),
        noise_step=0.05
    )
    random_hf = random_cfg.func(difficulty)
    print(f"Random uniform terrain shape: {random_hf.shape}, dtype: {random_hf.dtype}")
    print(f"Random uniform terrain min/max: {random_hf.min().item()}/{random_hf.max().item()}")

    # 2. 金字塔斜坡地形
    pyramid_slope_cfg = HfPyramidSlopedTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,
        slope_range=(0.1, 0.5),
        platform_width=2.0,
        inverted=False
    )
    pyramid_slope_hf = pyramid_slope_cfg.func(difficulty)
    print(f"\nPyramid sloped terrain shape: {pyramid_slope_hf.shape}")
    print(f"Pyramid sloped terrain min/max: {pyramid_slope_hf.min().item()}/{pyramid_slope_hf.max().item()}")

    # 3. 波浪地形
    wave_cfg = HfWaveTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,
        amplitude_range=(0.1, 0.5),
        num_waves=4
    )
    wave_hf = wave_cfg.func(difficulty)
    print(f"\nWave terrain shape: {wave_hf.shape}")
    print(f"Wave terrain min/max: {wave_hf.min().item()}/{wave_hf.max().item()}")

    # 4. 垫脚石地形
    stepping_stones_cfg = HfSteppingStonesTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,
        stone_width_range=(0.3, 0.8),
        stone_distance_range=(0.1, 0.5),
        stone_height_max=0.3,
        holes_depth=-0.5,
        platform_width=2.0
    )
    stepping_stones_hf = stepping_stones_cfg.func(difficulty)
    print(f"\nStepping stones terrain shape: {stepping_stones_hf.shape}")
    print(f"Stepping stones terrain min/max: {stepping_stones_hf.min().item()}/{stepping_stones_hf.max().item()}")

    # 5. 离散障碍物地形
    obstacles_cfg = HfDiscreteObstaclesTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,
        num_obstacles=15,
        obstacle_height_range=(0.2, 0.6),
        obstacle_width_range=(0.5, 0.9),
        obstacle_height_mode="choice",
        platform_width=2.0
    )
    obstacles_hf = obstacles_cfg.func(difficulty)
    print(f"\nDiscrete obstacles terrain shape: {obstacles_hf.shape}")
    print(f"Discrete obstacles terrain min/max: {obstacles_hf.min().item()}/{obstacles_hf.max().item()}")

    # 6. 金字塔阶梯地形
    stairs_cfg = HfPyramidStairsTerrainCfg(
        size=(8.0, 8.0),
        proportion=0.1,
        step_height_range=(0.1, 0.25),
        step_width=0.6,
        platform_width=2.0,
        inverted=False
    )
    stairs_hf = stairs_cfg.func(difficulty)
    print(f"\nPyramid stairs terrain shape: {stairs_hf.shape}")
    print(f"Pyramid stairs terrain min/max: {stairs_hf.min().item()}/{stairs_hf.max().item()}")

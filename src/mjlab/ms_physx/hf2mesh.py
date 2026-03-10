import torch
import trimesh
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


@dataclass
class HeightFieldConfig:
    """高度场配置参数"""
    # 地形的物理尺寸 (x轴长度, y轴长度)，单位：米
    size: Tuple[float, float] = (10.0, 10.0)
    # x/y方向的离散化精度（每个像素代表的实际物理长度），单位：米/像素
    horizontal_scale: float = 0.1
    # z方向的缩放比例（高度值的单位转换系数），单位：米/单位高度
    vertical_scale: float = 0.1
    # 边界宽度，用于避免地形边缘出现渲染瑕疵，单位：米
    border_width: float = 0.2
    # 坡度阈值（弧度），超过该阈值的坡面会被修正为垂直面，None表示不修正
    slope_threshold: Optional[float] = None
    # 计算设备（cpu/cuda），cuda可利用GPU加速计算
    device: str = "cpu"
    # 张量数据类型，float32兼顾精度和性能
    dtype: torch.dtype = torch.float32


def convert_height_field_to_mesh(
        height_field: torch.Tensor,
        horizontal_scale: float,
        vertical_scale: float,
        slope_threshold: Optional[float] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将高度场 Tensor 转换为三角形网格的顶点和三角形索引
    核心功能：把二维高度矩阵转换成可渲染的3D网格数据

    Args:
        height_field: 2D 高度场 Tensor (num_rows, num_cols)，每个值代表对应位置的高度
        horizontal_scale: x/y 方向的离散化精度（米/像素）
        vertical_scale: z 方向的缩放比例（米/单位高度）
        slope_threshold: 坡度阈值（弧度），超过则修正垂直面，None 表示不修正
        device: 计算设备（cpu/cuda）
        dtype: 数据类型

    Returns:
        vertices: 顶点 Tensor (num_vertices, 3)，每个元素为 (x, y, z) 三维坐标
        triangles: 三角形索引 Tensor (num_triangles, 3)，每个元素为顶点索引（构成三角形）
    """
    # 获取高度场的行列数（像素维度）
    num_rows, num_cols = height_field.shape
    # 将高度场张量转移到指定设备并转换数据类型，保证计算一致性
    height_field = height_field.to(device=device, dtype=dtype)

    # ===================== 第一步：创建网格的基础坐标 =====================
    # 生成y轴方向的坐标值（从0到最大物理长度，均匀分布）
    y = torch.linspace(0, (num_cols - 1) * horizontal_scale, num_cols, device=device, dtype=dtype)
    # 生成x轴方向的坐标值（从0到最大物理长度，均匀分布）
    x = torch.linspace(0, (num_rows - 1) * horizontal_scale, num_rows, device=device, dtype=dtype)
    # 生成二维网格坐标，indexing="xy"表示笛卡尔坐标系（行=x，列=y）
    yy, xx = torch.meshgrid(y, x, indexing="xy")  # 输出形状：(num_rows, num_cols)
    # 复制高度场数据，避免修改原始输入
    hf = height_field.clone()

    # ===================== 第二步：坡度修正（可选） =====================
    # 如果设置了坡度阈值，修正超过阈值的陡峭坡面，避免网格出现异常
    if slope_threshold is not None:
        # 根据缩放比例调整坡度阈值，将物理坡度转换为像素空间的高度差阈值
        slope_threshold_scaled = slope_threshold * horizontal_scale / vertical_scale

        # 初始化顶点移动量张量（记录每个像素需要移动的量）
        move_x = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)  # x方向移动量
        move_y = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)  # y方向移动量
        move_corners = torch.zeros((num_rows, num_cols), device=device, dtype=torch.int32)  # 对角线方向移动量

        # 沿x轴方向修正（上下相邻像素的高度差超过阈值）
        # 下一行比当前行高太多：当前行需要向右移动
        move_x[:-1, :] += (hf[1:, :] - hf[:-1, :]) > slope_threshold_scaled
        # 上一行比当前行高太多：当前行需要向左移动
        move_x[1:, :] -= (hf[:-1, :] - hf[1:, :]) > slope_threshold_scaled

        # 沿y轴方向修正（左右相邻像素的高度差超过阈值）
        # 右一列比当前列高太多：当前列需要向上移动
        move_y[:, :-1] += (hf[:, 1:] - hf[:, :-1]) > slope_threshold_scaled
        # 左一列比当前列高太多：当前列需要向下移动
        move_y[:, 1:] -= (hf[:, :-1] - hf[:, 1:]) > slope_threshold_scaled

        # 沿对角线方向修正（对角相邻像素的高度差超过阈值）
        # 右下像素比当前像素高太多：当前像素需要向对角线移动
        move_corners[:-1, :-1] += (hf[1:, 1:] - hf[:-1, :-1]) > slope_threshold_scaled
        # 左上像素比当前像素高太多：当前像素需要向对角线反方向移动
        move_corners[1:, 1:] -= (hf[:-1, :-1] - hf[1:, 1:]) > slope_threshold_scaled

        # 将移动量转换为浮点型，用于坐标计算
        move_x_float = move_x.to(dtype=dtype)
        move_y_float = move_y.to(dtype=dtype)
        move_corners_float = move_corners.to(dtype=dtype)

        # 应用顶点移动：优先x/y方向，无移动时再应用对角线移动
        xx += (move_x_float + move_corners_float * (move_x == 0).to(dtype=dtype)) * horizontal_scale
        yy += (move_y_float + move_corners_float * (move_y == 0).to(dtype=dtype)) * horizontal_scale

    # ===================== 第三步：生成网格顶点 =====================
    # 初始化顶点张量：每个像素对应一个顶点，共num_rows*num_cols个，每个顶点有x/y/z三个坐标
    vertices = torch.zeros((num_rows * num_cols, 3), device=device, dtype=dtype)
    # 将二维坐标展平为一维，赋值给顶点的x坐标
    vertices[:, 0] = xx.flatten()
    # 将二维坐标展平为一维，赋值给顶点的y坐标
    vertices[:, 1] = yy.flatten()
    # 将高度值展平并缩放，赋值给顶点的z坐标（高度）
    vertices[:, 2] = hf.flatten() * vertical_scale

    # ===================== 第四步：生成三角形索引 =====================
    # 计算三角形数量：每个像素格子生成2个三角形，共2*(行数-1)*(列数-1)个
    num_triangles = 2 * (num_rows - 1) * (num_cols - 1)
    # 初始化三角形索引张量：每个三角形包含3个顶点索引
    triangles = torch.zeros((num_triangles, 3), device=device, dtype=torch.int32)

    # 使用向量化操作生成索引（比循环高效）
    # 生成行索引范围（0到num_rows-2）
    i = torch.arange(num_rows - 1, device=device)
    # 生成列索引范围（0到num_cols-2）
    j = torch.arange(num_cols - 1, device=device)
    # 生成二维索引网格
    ii, jj = torch.meshgrid(i, j, indexing="xy")  # 形状：(num_rows-1, num_cols-1)

    # 计算每个格子的四个顶点索引（一维展平后的索引）
    idx0 = ii * num_cols + jj          # 左上角顶点 (i, j)
    idx1 = ii * num_cols + (jj + 1)    # 右上角顶点 (i, j+1)
    idx2 = (ii + 1) * num_cols + jj    # 左下角顶点 (i+1, j)
    idx3 = (ii + 1) * num_cols + (jj + 1)  # 右下角顶点 (i+1, j+1)

    # 将二维索引展平为一维数组，便于批量赋值
    idx0_flat = idx0.flatten()
    idx1_flat = idx1.flatten()
    idx2_flat = idx2.flatten()
    idx3_flat = idx3.flatten()

    # 填充第一个三角形（偶数索引）：左上角-右下角-右上角 (0,3,1)
    triangles[::2, 0] = idx0_flat
    triangles[::2, 1] = idx3_flat
    triangles[::2, 2] = idx1_flat

    # 填充第二个三角形（奇数索引）：左上角-左下角-右下角 (0,2,3)
    triangles[1::2, 0] = idx0_flat
    triangles[1::2, 1] = idx2_flat
    triangles[1::2, 2] = idx3_flat

    # 返回顶点坐标和三角形索引
    return vertices, triangles


def generate_mesh_from_height_field(
        height_field_func: Callable[[float, HeightFieldConfig], torch.Tensor],
        difficulty: float = 0.5,
        cfg: Optional[HeightFieldConfig] = None
) -> Tuple[trimesh.Trimesh, torch.Tensor]:
    """
    从高度场函数生成 trimesh 网格对象（全程使用 Tensor 计算）
    封装完整的地形生成流程：创建带边界的高度场 -> 转换为网格 -> 计算原点

    Args:
        height_field_func: 高度场生成函数，输入(难度值, 配置)，返回2D Tensor高度场
        difficulty: 难度参数（控制地形复杂度/高度），范围 [0, 1]
        cfg: 高度场配置，默认使用默认配置

    Returns:
        mesh: trimesh 网格对象（可直接可视化/导出）
        origin: 地形原点坐标 Tensor (x, y, z)，位于地形中心底部
    """
    # 使用默认配置如果未提供
    if cfg is None:
        cfg = HeightFieldConfig()

    # 验证边界宽度：必须大于等于水平缩放比例，否则边界无意义
    if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
        raise ValueError(
            f"边界宽度 ({cfg.border_width}) 必须大于等于水平缩放比例 ({cfg.horizontal_scale})"
        )

    # ===================== 第一步：计算像素尺寸（包含边界） =====================
    # 计算核心区域的像素宽度（物理尺寸 / 像素精度）
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    # 计算核心区域的像素长度
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # 计算边界的像素宽度（物理边界宽度 / 像素精度）
    border_pixels = int(cfg.border_width / cfg.horizontal_scale)

    # 初始化高度场数组（包含边界）：核心区域+上下左右边界
    heights = torch.zeros((width_pixels+2*border_pixels, length_pixels+2*border_pixels),
                          device=cfg.device, dtype=torch.int16)

    # ===================== 第二步：生成核心区域高度场 =====================
    # 临时创建仅包含核心区域的配置（用于高度场生成函数）
    core_cfg = HeightFieldConfig(
        size=cfg.size,
        horizontal_scale=cfg.horizontal_scale,
        vertical_scale=cfg.vertical_scale,
        device=cfg.device,
        dtype=cfg.dtype
    )

    # 调用高度场生成函数，生成核心区域的高度场
    core_heights = height_field_func(difficulty, core_cfg)
    # 确保核心高度场的设备和类型与整体一致
    core_heights = core_heights.to(device=cfg.device, dtype=torch.int16)

    # 将核心区域高度场放入包含边界的数组中（边界区域保持0高度）
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = core_heights

    # ===================== 第三步：转换为网格 =====================
    # 调用转换函数，生成顶点和三角形索引
    vertices, triangles = convert_height_field_to_mesh(
        heights,
        horizontal_scale=cfg.horizontal_scale,
        vertical_scale=cfg.vertical_scale,
        slope_threshold=cfg.slope_threshold,
        device=cfg.device,
        dtype=cfg.dtype
    )

    # ===================== 第四步：创建trimesh对象 =====================
    # trimesh不支持Tensor，转换为numpy数组
    vertices_np = vertices.cpu().numpy()
    triangles_np = triangles.cpu().numpy().astype(np.uint32)

    # 创建trimesh网格对象（可用于可视化、导出、碰撞检测等）
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=triangles_np)

    # ===================== 第五步：计算地形原点 =====================
    # 计算地形中心的x/y坐标（物理中心）
    center_x = cfg.size[0] / 2.0
    center_y = cfg.size[1] / 2.0

    # 计算中心区域的像素索引（中心周围2米范围）
    x1 = int((center_x - 1.0) / cfg.horizontal_scale)
    x2 = int((center_x + 1.0) / cfg.horizontal_scale)
    y1 = int((center_y - 1.0) / cfg.horizontal_scale)
    y2 = int((center_y + 1.0) / cfg.horizontal_scale)

    # 确保索引在有效范围内（防止越界）
    x1 = max(0, x1)
    x2 = min(width_pixels, x2)
    y1 = max(0, y1)
    y2 = min(length_pixels, y2)

    # 获取中心区域的高度值
    center_heights = heights[x1:x2, y1:y2]
    # 计算中心区域的最大高度作为原点的z坐标（保证原点在地形表面）
    origin_z = center_heights.max() * cfg.vertical_scale

    # 构造原点张量（中心底部坐标）
    origin = torch.tensor([center_x, center_y, origin_z],
                          device=cfg.device, dtype=cfg.dtype)

    return mesh, origin


# ------------------------------
# 示例：如何使用（全程 Tensor 计算）
# ------------------------------
if __name__ == "__main__":
    # 1. 定义一个基于 Tensor 的高度场生成函数（正弦波浪地形）
    def sine_wave_terrain(difficulty: float, cfg: HeightFieldConfig) -> torch.Tensor:
        """生成正弦波浪高度场（返回 Tensor）
        功能：生成基于正弦/余弦函数的波浪形地形，难度参数控制波浪幅度
        """
        # 计算核心区域的像素尺寸（+1是因为linspace包含首尾）
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1

        # 创建x轴网格坐标（物理坐标，从0到地形尺寸）
        x = torch.linspace(0, cfg.size[0], width_pixels,
                           device=cfg.device, dtype=cfg.dtype)
        # 创建y轴网格坐标
        y = torch.linspace(0, cfg.size[1], length_pixels,
                           device=cfg.device, dtype=cfg.dtype)
        # 生成二维网格
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # 生成正弦波浪地形
        amplitude = 0.5 * difficulty  # 波浪幅度：难度0→0米，难度1→0.5米
        frequency = 2.0  # 频率：每米2个波浪
        # x方向正弦波 + y方向余弦波，组合成波浪地形
        heights = amplitude * torch.sin(2 * torch.pi * frequency * xx / cfg.size[0]) + \
                  amplitude * torch.cos(2 * torch.pi * frequency * yy / cfg.size[1])

        # 转换为int16类型（整数存储节省内存，需除以垂直缩放比例）
        return (heights / cfg.vertical_scale).to(dtype=torch.int16)


    # 2. 配置参数（自动检测CUDA加速）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HeightFieldConfig(
        size=(20.0, 20.0),  # 地形尺寸 20x20 米
        horizontal_scale=0.05,  # 5 厘米/像素的精度（越高精度越细腻）
        vertical_scale=0.05,  # 5 厘米/单位高度
        border_width=0.1,  # 10 厘米边界（防止边缘异常）
        # 坡度阈值：60度（转换为弧度后计算正切值），超过则修正
        slope_threshold=torch.tan(torch.tensor(np.deg2rad(60), dtype=torch.float32)),
        device=device,  # 使用检测到的设备（CPU/GPU）
        dtype=torch.float32  # 浮点精度
    )

    # 3. 生成网格（全程使用 Tensor 计算）
    mesh, origin = generate_mesh_from_height_field(
        height_field_func=sine_wave_terrain,  # 高度场生成函数
        difficulty=0.8,  # 高难度 = 更大的波浪幅度
        cfg=config  # 配置参数
    )

    # 4. 输出信息并可视化
    print(f"计算设备：{device}")
    print(f"生成的网格：{len(mesh.vertices)} 个顶点，{len(mesh.faces)} 个三角形")
    print(f"地形原点：{origin.cpu().numpy()}")

    # 显示网格（需要安装 pycollada 依赖）
    mesh.show()

    # 5. 保存网格到文件（可选）
    # mesh.export("terrain_tensor.obj")

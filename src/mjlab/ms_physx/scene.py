# -*- coding: utf-8 -*-
"""
仿真场景管理模块
负责创建和管理mozisim仿真环境，包括地形加载、机器人/物体添加、场景重置等核心功能
"""

# 从mozisim仿真核心库导入Scene类，并重命名为MoziScene避免命名冲突
from mozisim.core.api.scene import Scene as MoziScene
from ms_lab.ms_physx.terrain_impoter import TerrainImporter
from ms_lab.ms_physx.object import Object


class Scene:
    """
    仿真场景管理类

    核心功能：
    - 创建并初始化mozisim仿真场景
    - 加载地形数据到仿真环境
    - 批量添加机器人模型到多环境仿真场景
    - 创建仿真物体实例
    - 重置指定环境的仿真状态

    Attributes:
        cfg: 配置对象，包含场景相关配置（如环境数量num_envs、地形配置terrain等）
        device: 设备对象，指定运行仿真的硬件设备（CPU/GPU）
        num_envs: 并行仿真的环境数量，从配置对象中读取
        scene: mozisim核心场景实例，负责底层仿真场景管理
        _terrain: 地形导入器实例，私有属性，存储已加载的地形数据
    """

    def __init__(self, cfg, device):
        """
        初始化场景管理器

        Args:
            cfg: 配置对象
                关键配置项：
                - num_envs: 并行仿真的环境数量
                - terrain: 地形相关配置参数
            device: 设备对象，指定仿真运行的硬件设备（如torch.device('cuda:0')）
        """
        # 保存配置和设备信息
        self.cfg = cfg
        self.device = device

        # 从配置中获取多环境并行仿真的环境数量
        self.num_envs = self.cfg.num_envs

        # 初始化mozisim核心场景对象（底层仿真场景）
        self.scene = MoziScene()

        # 初始化地形属性（私有属性，通过terrain属性访问）
        self._terrain = None

    @property
    def terrain(self):
        """
        地形属性访问器

        Returns:
            TerrainImporter: 已加载的地形导入器实例，若无则返回None
        """
        return self._terrain

    def add_terrain(self):
        """
        向仿真场景添加默认地面地形

        功能说明：
        - 创建地形导入器实例
        - 加载配置中指定的地形数据到仿真场景
        注：原代码中注释掉的add_default_ground_plane方法为备选方案
        """
        # 初始化地形导入器，加载配置中定义的地形数据
        self._terrain = TerrainImporter(self.cfg.terrain, self.device)

        # 备选方案：添加默认地面平面（已注释，保留供参考）
        # self.scene.add_default_ground_plane(prim_path="/World/GroundPlane")

    def add_robot(self, ent_name, ent_cfg):
        """
        批量添加机器人模型到多环境仿真场景

        为每个仿真环境创建一个独立的机器人实例，确保每个机器人在USD场景树中有唯一路径

        Args:
            ent_name: 机器人基础名称（如"robot"），会自动添加环境索引后缀
            ent_cfg: 机器人实体配置对象
                关键配置项：
                - asset_file: 机器人USD模型文件的路径

        Returns:
            list[str]: 成功添加的机器人在USD场景树中的路径列表
                       示例：["/World/robot_0", "/World/robot_1"]
        """
        # 存储所有成功添加的机器人USD场景路径
        robot_prim_paths = []

        # 按环境数量循环，为每个环境创建一个机器人实例
        for env_idx in range(self.num_envs):
            # 生成带环境索引的机器人名称，确保唯一性
            robot_name = f"{ent_name}_{env_idx}"
            # 定义机器人在USD场景树中的唯一路径
            robot_prim_path = f"/World/{robot_name}"

            # 向仿真场景中添加机器人模型引用（基于USD文件）
            robot_prim = self.scene.add_reference_to_stage(
                usd_path=ent_cfg.asset_file,  # 机器人USD模型文件路径
                prim_path=robot_prim_path,  # 机器人在场景中的唯一路径
            )

            # 验证机器人是否成功添加
            if robot_prim and robot_prim.IsValid():
                robot_prim_paths.append(robot_prim_path)
            else:
                # 打印错误提示，包含具体环境索引便于调试
                print(f"【错误】环境 {env_idx} 中添加机器人 {robot_name} 失败！")

        # 返回所有成功添加的机器人路径列表
        return robot_prim_paths

    def add_object(self, object_name, object_cfg):
        """
        创建仿真物体实例

        Args:
            object_name: 物体名称，用于标识该物体实例
            object_cfg: 物体配置对象，包含物体的物理属性、模型路径等配置

        Returns:
            Object: 创建完成的仿真物体实例
        """
        # 初始化并返回物体实例
        object_instance = Object(object_name, object_cfg, self.device)
        return object_instance

    def reset(self, env_ids):
        """
        重置指定环境的仿真状态

        待实现功能：
        - 重置指定环境中机器人的位姿
        - 重置物体的位置/状态
        - 重置地形交互状态等

        Args:
            env_ids: list[int]，需要重置的环境ID列表（如[0, 2, 3]）
        """
        # 暂未实现具体重置逻辑，预留接口
        pass

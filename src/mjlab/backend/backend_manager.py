
class BackendManager:
    """
    后端管理器类
    统一管理不同物理后端（如mujoco/physx）的实例化和接口调用，提供统一的访问入口，
    屏蔽不同后端的实现差异，降低上层代码与具体物理后端的耦合度。
    """
    def __init__(self, backend_type, render_mode, device):
        """
        初始化后端管理器

        参数:
            backend_type (str): 物理后端类型，支持 "mujoco" / "physx"
            render_mode (str): 渲染模式（如"rgb_array" / "human"等），主要用于physx后端
            device (str/torch.device): 计算设备，如 "cpu" / "cuda:0"
        """
        self.backend_type = backend_type  # 保存后端类型
        self.device = device              # 保存计算设备

        # 根据后端类型实例化对应的物理后端
        if backend_type == "mujoco":
            # 延迟导入mujoco相关模块，避免不必要的依赖加载
            from ms_lab.backend.mujoco_backend import MujocoBackend
            import mujoco
            self.backend = MujocoBackend(device=self.device)
        elif backend_type == "physx":
            # 延迟导入physx相关模块
            from ms_lab.backend.physx_backend import PhysxBackend
            self.backend = PhysxBackend(render_mode, device)

    @property
    def num_envs(self):
        """
        属性方法：获取环境数量
        返回值:
            int: 当前后端中并行的环境数量
        """
        return self.backend.num_envs

    def write_data_to_sim(self):
        """将数据写入仿真环境（如机器人状态、控制指令等）"""
        self.backend.write_data_to_sim()

    def get_all_robot(self):
        """
        获取所有机器人实例
        返回值:
            list: 包含所有机器人对象的列表
        """
        return self.backend.get_all_robots()  # 注：原代码方法名拼写不一致（get_all_robot/get_all_robots），建议统一

    def get_terrain(self):
        """
        获取地形实例
        返回值:
            Terrain对象: 当前仿真环境的地形对象
        """
        return self.backend.get_terrain()

    def get_env_origins(self):
        """
        获取每个环境的原点坐标
        返回值:
            np.ndarray/torch.Tensor: 形状为(num_envs, 3)的坐标数组
        """
        return self.backend.get_env_origins()

    def _init_scene(self, cfg):
        """
        初始化仿真场景（内部方法）
        参数:
            cfg (dict/Config): 场景配置参数（如地形、机器人、灯光等配置）
        """
        self.backend._init_scene(cfg)

    def _init_sim(self, cfg):
        """
        初始化仿真器（内部方法）
        参数:
            cfg (dict/Config): 仿真器配置参数（如时间步长、并行环境数等）
        """
        self.backend._init_sim(cfg)

    def get_robot(self, asset_name):
        """
        根据资产名称获取指定机器人实例
        参数:
            asset_name (str): 机器人资产名称（如"walker" / "arm"）
        返回值:
            Robot对象: 对应名称的机器人实例
        """
        return self.backend.get_robot(asset_name)

    def expand_model_fields(self, domain_randomization_fields):
        """
        扩展模型字段（用于域随机化）
        参数:
            domain_randomization_fields (list/dict): 需要随机化的字段列表/配置
        """
        self.backend.expand_model_fields(domain_randomization_fields)

    def create_graph(self):
        """创建计算图（主要用于GPU加速的物理后端，如physx）"""
        self.backend.create_graph()

    def random_field(self, env_ids, field, ranges, distribution, operation, asset_cfg, axes):
        """
        对指定字段进行域随机化
        参数:
            env_ids (list/int): 需要随机化的环境ID（单ID或ID列表）
            field (str): 要随机化的字段名（如"mass" / "friction" / "stiffness"）
            ranges (tuple/list): 随机值范围，如 (0.8, 1.2)
            distribution (str): 分布类型（如"uniform" / "normal"）
            operation (str): 操作类型（如"multiply" / "add"）
            asset_cfg (dict): 资产配置信息
            axes (list/int): 作用轴（如[0,1,2]表示xyz轴）
        """
        self.backend.random_field(
            env_ids,
            field,
            ranges,
            distribution,
            operation,
            asset_cfg,
            axes
        )

    def get_stiffness(self, ctrl_ids):
        """
        获取指定控制ID的刚度值
        参数:
            ctrl_ids (list/int): 控制关节ID（单ID或ID列表）
        返回值:
            np.ndarray/torch.Tensor: 对应ID的刚度值数组
        """
        return self.backend.get_stiffness(ctrl_ids)

    def forward(self):
        """执行仿真前向计算（更新动力学状态，不推进时间步）"""
        self.backend.forward()

    def step(self):
        """推进仿真时间步（执行一次物理仿真迭代）"""
        self.backend.step()

    def update(self, dt):
        """
        更新仿真状态（通常与时间步相关）
        参数:
            dt (float): 时间步长（如0.001秒）
        """
        self.backend.update(dt)

    def reset(self, env_ids):
        """
        重置指定环境
        参数:
            env_ids (list/int): 需要重置的环境ID（单ID或ID列表）
        """
        self.backend.reset(env_ids)

    def get_dampling(self, ctrl_ids):
        """
        获取指定控制ID的阻尼值（注：原代码拼写错误，应为damping）
        参数:
            ctrl_ids (list/int): 控制关节ID（单ID或ID列表）
        返回值:
            np.ndarray/torch.Tensor: 对应ID的阻尼值数组
        """
        return self.backend.get_dampling(ctrl_ids)

    def render(self):
        """
        渲染仿真画面（仅支持mozi后端，原代码条件判断可能存在笔误，应为mujoco/physx）
        返回值:
            np.ndarray: 渲染的图像数组（如形状为(H, W, 3)的RGB图像），无渲染时返回None
        """
        if self.backend_type == "mozi":  # 疑似笔误：应为"mujoco"或"physx"
            return self.backend.render()

    def close(self):
        """关闭仿真后端，释放资源（仅支持mozi后端，疑似笔误）"""
        if self.backend_type == "mozi":  # 疑似笔误：应为"mujoco"或"physx"
            self.backend.close()

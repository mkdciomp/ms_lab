# 导入必要的内置模块
import threading  # 用于线程安全的锁机制，保护共享的按键状态
import time  # 用于时间戳、按键持续时长计算和程序休眠
import numpy as np  # 用于向量长度计算和数值处理

# 导入第三方库（需要先安装：pip install pynput）
from pynput import keyboard  # pynput库的键盘监听模块，用于捕获键盘事件

# ======================== 全局模拟变量（实际使用时替换为真实关节数据） ========================
# 模拟"站起来"的关节位置数据（示例）
stand_up_joint_pos = [0.0, 1.57, -1.57, 0.0, 1.57, -1.57]
# 模拟"趴下"的关节位置数据（示例）
stand_down_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ======================== 机器人键盘控制器类 ========================
class PynputListener:
    def __init__(self, max_vector_value: float = 5.0, sensitivity: float = 1.0, print_interval: float = 0.1,
                 fps: int = 60):
        """
        初始化机器人键盘控制器

        参数说明：
        - max_vector_value: 方向向量的最大值（限制向量长度）
        - sensitivity: 按键灵敏度（按键时长的放大系数）
        - print_interval: 状态打印的时间间隔（秒）
        - fps: 控制器更新帧率（次/秒）
        """
        self.max_vector = max_vector_value
        self.sensitivity = sensitivity
        self.print_interval = print_interval
        self.fps = fps

        # 按键状态（线程安全）：记录按键是否按下、按下起始时间
        self.key_states = {
            'w': {'pressed': False, 'start_time': 0.0},
            'a': {'pressed': False, 'start_time': 0.0},
            's': {'pressed': False, 'start_time': 0.0},
            'd': {'pressed': False, 'start_time': 0.0},
            'enter': {'pressed': False},
            'up': {'pressed': False},
            'down': {'pressed': False},
        }
        self.lock = threading.Lock()  # 线程锁，保证多线程下按键状态的安全访问

        self.is_recording = False  # 是否正在录制向量数据
        self.recorded_data = []  # 录制的向量数据列表
        self.last_print_time = 0  # 上一次打印状态的时间
        self.running = False  # 控制器是否运行

        self.current_x, self.current_y = 0.0, 0.0  # 当前向量的x/y分量
        self.mode = "command"  # 工作模式：command（命令）/ vector（向量）
        self.current_command = "等待命令"  # 当前执行的命令
        self.current_joint_pos = None  # 当前目标关节位置

        self.listener = None  # 键盘监听器实例

    def _on_press(self, key):
        """键盘按下事件处理函数（内部方法）"""
        with self.lock:  # 加锁保证线程安全
            try:
                # 统一按键标识格式
                key_str = key.char if hasattr(key, 'char') else str(key)

                # ESC键：退出控制器
                if key == keyboard.Key.esc:
                    self.running = False
                    return False  # 返回False让系统也收到ESC（保证Viewer也能关闭）

                # Enter键：切换工作模式
                if key == keyboard.Key.enter:
                    if not self.key_states['enter']['pressed']:
                        self.mode = "command" if self.mode == "vector" else "vector"
                        mode_name = "命令模式" if self.mode == "command" else "向量控制模式"
                        print(f"\n已切换到{mode_name}")
                        if self.mode == "vector":
                            self.current_command = None
                            self.current_joint_pos = None
                        else:
                            self.current_command = "等待命令"
                            self.current_joint_pos = None
                        self.key_states['enter']['pressed'] = True

                # 上方向键：发送"站起来"命令
                elif key == keyboard.Key.up:
                    self.key_states['up']['pressed'] = True
                    if self.mode == "command":
                        self.current_command = "站起来"
                        self.current_joint_pos = stand_up_joint_pos
                        print(f"发送命令: {self.current_command} | 关节位置已更新")

                # 下方向键：发送"趴下"命令
                elif key == keyboard.Key.down:
                    self.key_states['down']['pressed'] = True
                    if self.mode == "command":
                        self.current_command = "趴下"
                        self.current_joint_pos = stand_down_joint_pos
                        print(f"发送命令: {self.current_command} | 关节位置已更新")

                # WASD键：向量控制（记录按下起始时间）
                elif hasattr(key, 'char') and key.char in ['w', 'a', 's', 'd']:
                    if not self.key_states[key.char]['pressed']:
                        self.key_states[key.char]['pressed'] = True
                        self.key_states[key.char]['start_time'] = time.time()

                # P键：开始录制向量数据
                elif hasattr(key, 'char') and key.char == 'p':
                    if not self.is_recording:
                        self.is_recording = True
                        self.recorded_data = []
                        print("开始录制向量数据...")

                # O键：停止录制向量数据
                elif hasattr(key, 'char') and key.char == 'o':
                    if self.is_recording:
                        self.is_recording = False
                        print(f"\n结束录制 | 录制数据量: {len(self.recorded_data)} 条")

            except Exception as e:
                print(f"[pynput错误] {e}")

    def _on_release(self, key):
        """键盘释放事件处理函数（内部方法）"""
        with self.lock:  # 加锁保证线程安全
            try:
                # 重置对应按键的按下状态
                if key == keyboard.Key.enter:
                    self.key_states['enter']['pressed'] = False
                elif key == keyboard.Key.up:
                    self.key_states['up']['pressed'] = False
                elif key == keyboard.Key.down:
                    self.key_states['down']['pressed'] = False
                elif hasattr(key, 'char') and key.char in ['w', 'a', 's', 'd']:
                    self.key_states[key.char]['pressed'] = False
            except Exception as e:
                pass

    def _get_press_duration(self, key):
        """获取指定按键的持续按下时间（秒）"""
        with self.lock:
            if not self.key_states[key]['pressed']:
                return 0.0
            return time.time() - self.key_states[key]['start_time']

    def _calculate_vector(self):
        """计算当前的方向向量（x,y）"""
        # 获取各按键的持续按下时间并乘以灵敏度
        w = self._get_press_duration('w') * self.sensitivity
        s = self._get_press_duration('s') * self.sensitivity
        a = self._get_press_duration('a') * self.sensitivity
        d = self._get_press_duration('d') * self.sensitivity

        # 限制单个方向的最大值
        w = min(w, self.max_vector)
        s = min(s, self.max_vector)
        a = min(a, self.max_vector)
        d = min(d, self.max_vector)

        # 计算向量分量：x（左右）=d-a，y（前后）=s-w
        x = d - a
        y = s - w

        # 归一化向量长度（不超过max_vector）
        vec_len = np.sqrt(x ** 2 + y ** 2) if (x != 0 or y != 0) else 0
        if vec_len > self.max_vector:
            scale = self.max_vector / vec_len
            x *= scale
            y *= scale

        # 保留两位小数返回
        return round(x, 2), round(y, 2)

    def _record_vector(self):
        """录制当前向量数据（内部方法）"""
        if self.is_recording and self.mode == "vector":
            self.recorded_data.append((self.current_x, self.current_y, time.time(), "vector"))

    def _print_status(self):
        """打印当前控制器状态（按指定时间间隔）"""
        if self.print_interval <= 0:
            return
        now = time.time()
        if now - self.last_print_time > self.print_interval:
            # 组装状态信息
            mode_info = "【向量模式】" if self.mode == "vector" else "【命令模式】"
            record_status = "【录制中】" if self.is_recording else ""

            if self.mode == "vector":
                status_text = f"方向向量: ({self.current_x:.2f}, {self.current_y:.2f}) {mode_info} {record_status}"
            else:
                joint_info = "已更新" if self.current_joint_pos is not None else "未设置"
                status_text = f"当前命令: {self.current_command} | 关节位置: {joint_info} {mode_info} {record_status}"

            # 打印状态（\r实现覆盖打印）
            print(status_text.ljust(80), end="\r")
            self.last_print_time = now

    def update(self):
        """控制器主更新循环"""
        if not self.running:
            return
        # 向量模式下更新方向向量
        if self.mode == "vector":
            self.current_x, self.current_y = self._calculate_vector()
        # 录制向量数据（如果需要）
        self._record_vector()
        # 打印状态信息
        self._print_status()
        # 控制更新帧率
        time.sleep(1.0 / self.fps)

    def start(self):
        """启动控制器"""
        self.running = True
        # 创建并启动键盘监听器
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=True  # 抑制系统默认的键盘事件（防止按键输入到其他程序）
        )
        self.listener.start()
        # 打印启动信息
        print("=" * 80)
        print("机器人控制器 V2.0（pynput 稳定版）")
        print("✅ pynput监听器已启动（suppress=True）")
        print("操作说明:")
        print("  - Enter: 切换模式（命令 ↔ 向量）")
        print("  - 向量模式: WASD 控制方向, P 录制, O 停止录制")
        print("  - 命令模式: ↑ 站起来, ↓ 趴下")
        print("  - ESC: 退出")
        print("=" * 80)

    def stop(self):
        """停止控制器"""
        self.running = False
        if self.listener:
            self.listener.stop()

    def cleanup(self):
        """资源清理"""
        self.stop()
        print("\n资源已释放")

    def get_current_state(self):
        """获取当前控制器状态"""
        return {
            "mode": self.mode,
            "vector": (self.current_x, self.current_y) if self.mode == "vector" else None,
            "command": self.current_command if self.mode == "command" else None,
            "joint_position": self.current_joint_pos.copy() if self.current_joint_pos is not None else None,
            "is_recording": self.is_recording,
            "recorded_count": len(self.recorded_data)
        }

    def get_recorded_vectors(self):
        """获取录制的向量数据副本"""
        return self.recorded_data.copy()

    def get_joint_position_by_command(self, command):
        """根据命令获取对应的关节位置"""
        if command == "站起来":
            return stand_up_joint_pos.copy()
        elif command == "趴下":
            return stand_down_joint_pos.copy()
        else:
            return None


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # 1. 创建控制器实例（可自定义参数）
    controller = PynputListener(
        max_vector_value=5.0,  # 向量最大值
        sensitivity=2.0,  # 灵敏度提高到2倍
        print_interval=0.1,  # 每0.1秒打印一次状态
        fps=60  # 60帧/秒更新
    )

    try:
        # 2. 启动控制器
        controller.start()

        # 3. 主循环：持续更新控制器状态
        while controller.running:
            controller.update()

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断
        print("\n\n程序被用户中断")
    finally:
        # 4. 清理资源
        controller.cleanup()

        # 5. 示例：打印录制的数据（如果有）
        recorded_data = controller.get_recorded_vectors()
        if recorded_data:
            print(f"\n录制的前5条向量数据：")
            for i, data in enumerate(recorded_data[:5]):
                print(f"  第{i + 1}条: 向量({data[0]:.2f}, {data[1]:.2f}) | 时间戳: {data[2]:.2f}")

        # 6. 示例：获取最后一次的状态
        final_state = controller.get_current_state()
        print(f"\n最终状态: {final_state}")

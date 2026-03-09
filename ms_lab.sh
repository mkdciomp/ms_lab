#!/bin/bash
# Mozi引导器（适配手动激活Conda/uv环境）
# 脚本用途：用于正确配置MoziSim仿真环境的依赖路径，然后执行指定的Python程序
# 使用方法：在终端执行 ./ms_lab.sh your_program.py 参数1 参数2...

# ======================== 核心路径配置区 ========================
# 获取当前脚本所在的绝对目录（解决相对路径可能导致的问题）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 定义mozisim安装目录的初始路径（相对于当前脚本）
MOZISIM_INSTALL_DIR="$SCRIPT_DIR/../mozisim_install"
# 将安装目录路径转换为绝对路径（确保路径准确性）
MOZISIM_INSTALL_DIR="$(cd "$MOZISIM_INSTALL_DIR" && pwd)"
# 定义mozisim的父目录（后续用于Python路径配置）
MOZISIM_PARENT_DIR="$MOZISIM_INSTALL_DIR"
# 定义mozisim的库文件目录（存放so共享库）
MOZISIM_LIB_DIR="$MOZISIM_INSTALL_DIR/lib"
# 定义OpenUSD的二进制文件目录
OPENUSD_BIN_DIR="$MOZISIM_INSTALL_DIR/openusd/bin"
# 定义PhysX物理引擎的二进制文件目录（Linux x86_64发布版）
PHYSX_BIN_DIR="$MOZISIM_INSTALL_DIR/physx/bin/linux.x86_64/release"

# 设置Mozi资源文件路径（UI相关的静态资源）
export MOZI_RESOURCES_PATH="$SCRIPT_DIR/../mozi_ui/assets/"

# 定义USD库目录的候选路径列表（适配不同安装方式的路径差异）
# 修复原脚本中可能缺失的USD库路径查找逻辑
OPENUSD_LIB_CANDIDATES=(
    "$MOZISIM_INSTALL_DIR/openusd/lib"          # 常见的lib目录
    "$MOZISIM_INSTALL_DIR/openusd/lib64"       # 64位系统的lib64目录
    "$MOZISIM_INSTALL_DIR/openusd/lib/linux.x86_64"  # 按系统架构划分的目录
)

# 自动遍历候选路径，查找存在USD核心库的目录
OPENUSD_LIB_DIR=""
for CANDIDATE in "${OPENUSD_LIB_CANDIDATES[@]}"; do
    # 检查是否存在USD核心库文件（两种常见命名）
    if [ -f "$CANDIDATE/libusd.so" ] || [ -f "$CANDIDATE/libusd_usd.so" ]; then
        OPENUSD_LIB_DIR="$CANDIDATE"  # 找到有效路径后赋值
        break  # 找到后立即退出循环，提升效率
    fi
done

# 定义USD的Python模块目录（基于找到的USD库路径）
OPENUSD_PYTHON_DIR="$OPENUSD_LIB_DIR/python"

# ======================== 参数检查区 ========================
# 检查用户是否传入了要执行的Python脚本路径
if [ -z "$1" ]; then
    echo "Error: Please provide the path of the Python script to be executed"
    echo "Usage: $0 <python_script> [script_args...]"
    exit 1  # 参数缺失，退出脚本并返回错误码1
fi

# 提取第一个参数作为要执行的Python脚本路径
SCRIPT_PATH="$1"
shift  # 将后续参数左移（去掉第一个参数，保留脚本的入参）

# 检查指定的Python脚本文件是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The specified Python script '$SCRIPT_PATH' does not exist."
    exit 1  # 文件不存在，退出脚本并返回错误码1
fi

# ======================== 依赖验证区 ========================
# 验证mozisim Python包是否存在（确保安装完成）
if [ ! -d "$MOZISIM_PARENT_DIR/mozisim" ]; then
    echo "Error: mozisim Python package not found in $MOZISIM_PARENT_DIR"
    exit 1  # 核心包缺失，退出脚本并返回错误码1
fi

# 验证USD库目录是否成功找到
if [ -z "$OPENUSD_LIB_DIR" ]; then
    echo "Error: USD library directory not found! Tried candidates:"
    # 列出所有尝试过的候选路径，方便用户排查
    for CANDIDATE in "${OPENUSD_LIB_CANDIDATES[@]}"; do echo "  - $CANDIDATE"; done
    echo "Please confirm the USD library path in mozisim_install/lib/openusd"
    exit 1  # USD库缺失，退出脚本并返回错误码1
fi

# 验证USD的lux模块库（非致命错误，仅警告）
# 允许软链接（-L），提升路径适配灵活性
if [ ! -f "$OPENUSD_LIB_DIR/libusd_usdLux.so" ] && [ ! -L "$OPENUSD_LIB_DIR/libusd_usdLux.so" ]; then
    echo "Warning: libusd_usdLux.so not found in $OPENUSD_LIB_DIR, but found other USD libraries"
    echo "Trying to proceed with available USD libraries..."
fi

# 验证mozisim后端核心库（非致命错误，仅警告）
if [ ! -f "$MOZISIM_LIB_DIR/libmozisim_backend.so" ]; then
    echo "Warning: libmozisim_backend.so not found in $MOZISIM_LIB_DIR"
    echo "Please confirm the shared library path and try again"
fi

# ======================== 环境变量配置区 ========================
# 定义项目根目录（相对于当前脚本）并转换为绝对路径
PROJECT_ROOT="$SCRIPT_DIR/../"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

# 配置PYTHONPATH环境变量（Python模块搜索路径）
# 优先级：mozisim安装目录 > mozisim库目录 > USD Python目录 > 项目根目录 > 原有路径
export PYTHONPATH="$MOZISIM_PARENT_DIR:$MOZISIM_LIB_DIR:$OPENUSD_PYTHON_DIR:$PROJECT_ROOT:$PYTHONPATH"

# 配置LD_LIBRARY_PATH环境变量（Linux共享库搜索路径）
# 优先级：mozisim库目录 > USD库目录 > PhysX库目录 > 原有路径
export LD_LIBRARY_PATH="$MOZISIM_LIB_DIR:$OPENUSD_LIB_DIR:$PHYSX_BIN_DIR:$LD_LIBRARY_PATH"

# 配置系统可执行文件路径（仅当USD二进制目录存在时）
[ -d "$OPENUSD_BIN_DIR" ] && export PATH="$OPENUSD_BIN_DIR:$PATH"

# ======================== 环境信息打印区 ========================
# 获取当前环境中python3的实际路径（若激活Conda，会显示Conda环境的Python）
PYTHON_PATH=$(which python3)
echo "========================================"
echo "当前使用的Python3路径: $PYTHON_PATH"
# 显示当前激活的Conda环境名称（grep '*'找到带星标的激活环境）
echo "Conda环境状态: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "mozisim root dir: $MOZISIM_PARENT_DIR"
echo "mozisim lib dir: $MOZISIM_LIB_DIR"
echo "Found USD lib dir: $OPENUSD_LIB_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "========================================"

# ======================== 脚本执行区 ========================
# 打印要执行的Python脚本路径
echo "Executing Python script: $SCRIPT_PATH"
# 使用当前环境的Python执行指定脚本，并传递所有后续参数
"$PYTHON_PATH" "$SCRIPT_PATH" "$@"

# 获取脚本执行的退出码
EXIT_CODE=$?
# 以相同的退出码退出当前bash脚本（保持执行结果一致性）
exit $EXIT_CODE

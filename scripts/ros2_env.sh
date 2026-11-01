# ROS 2 用ターミナルの初期化。使い方: source scripts/ros2_env.sh
REBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# LeRobot (conda) と ROS 2 (システム python) は同じターミナルで混ぜない。
if [ -n "${CONDA_PREFIX:-}" ] && ! declare -F conda >/dev/null; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh" || return 1
fi
while [ -n "${CONDA_PREFIX:-}" ]; do
    conda deactivate || return 1
done
source /opt/ros/humble/setup.bash
source "$REBOT_ROOT/rebotarm_ros2/install/setup.bash"
# FastDDS 共有メモリのロック残り対策 (公式 FAQ)
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
# 同一ネットワーク上の他の ROS 2 機器と混線しないよう固定
export ROS_DOMAIN_ID=42
# Use the project-local ABI build; the installed pip package stays unchanged.
# Roll back with REBOT_USE_STOCK_MOTORBRIDGE=1 before sourcing this script.
if [ "${REBOT_USE_STOCK_MOTORBRIDGE:-0}" = 1 ]; then
    unset MOTORBRIDGE_LIB
elif [ -z "${MOTORBRIDGE_LIB:-}" ]; then
    rebot_local_motorbridge="$REBOT_ROOT/third_party/motorbridge/target/release/libmotor_abi.so"
    if [ ! -r "$rebot_local_motorbridge" ]; then
        echo "motorbridge local build missing: run bash scripts/build_motorbridge_local.sh" >&2
        unset rebot_local_motorbridge
        return 1
    fi
    export MOTORBRIDGE_LIB="$rebot_local_motorbridge"
    unset rebot_local_motorbridge
fi
echo "ROS 2 Humble + rebotarm_ros2 loaded (python: $(which python3))"
echo "motorbridge ABI: ${MOTORBRIDGE_LIB:-installed package (stock)}"

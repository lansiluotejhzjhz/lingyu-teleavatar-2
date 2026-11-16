# TeleAvatar 2 ROS 2 环境（文档 2.4 / 2.6.2）。用法：source env.sh
export ROBOT_IP="${ROBOT_IP:-192.168.31.244}"
export ROS_DOMAIN_ID=29
export ROS_DISTRO=humble
# 文档要求使用 Humble 默认的 Fast DDS，清除其他工作区可能设置的 RMW
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash
# zenoh-bridge-ros2dds v1.9.0 放在本目录的 bin/ 下（1.10.x 需要 glibc 2.38，Ubuntu 22.04 跑不了）
_TA2_BIN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/bin"
case ":$PATH:" in *":$_TA2_BIN:"*) ;; *) export PATH="$_TA2_BIN:$PATH" ;; esac

"""解析配置文件中的 ROS 包路径。"""

import os
import re
from functools import lru_cache

import rospkg
import yaml


_ROS_FIND_PATTERN = re.compile(r"\$\(find\s+([A-Za-z0-9_][A-Za-z0-9_-]*)\)")


@lru_cache(maxsize=None)
def _ros_package_path(package_name):
    return rospkg.RosPack().get_path(package_name)


def resolve_ros_substitutions(value):
    """解析字符串中的 ``$(find <package>)`` 表达式。"""
    if not isinstance(value, str):
        return value

    def replace(match):
        package_name = match.group(1)
        try:
            return _ros_package_path(package_name)
        except rospkg.ResourceNotFound:
            raise FileNotFoundError(
                "ROS 包不存在，无法解析路径: {}".format(package_name)
            )

    return _ROS_FIND_PATTERN.sub(replace, os.path.expandvars(os.path.expanduser(value)))


def resolve_path(value):
    """解析 ROS 包路径表达式并返回绝对路径。"""
    if value is None:
        return None
    return os.path.abspath(resolve_ros_substitutions(value))


def load_yaml_config(config_path):
    """读取 YAML，并解析其中的 ROS 包路径表达式。"""
    resolved_config_path = resolve_path(config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as config_file:
        content = resolve_ros_substitutions(config_file.read())
    return yaml.safe_load(content) or {}

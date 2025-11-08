"""配置文件读取工具。

封装成独立模块方便复用，同时也便于后期扩展（例如增加缓存、模式校验等）。
"""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件并返回字典。

    Tips for beginners:
    - 如果读取失败，请先检查路径是否正确（可以结合 `cli/_resolve_config`）。
    - 若需要指定编码，可在此函数中扩展 `encoding` 参数。
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


"""配置文件可用性测试。

目标：帮助新同学快速确认配置中提到的路径是否存在，避免运行流水线前就出错。
运行 `python -m pytest` 时会自动执行这些检查。
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict:
    """读取 YAML 并返回字典，测试里不需要额外封装。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path_str: str) -> Path:
    """允许在配置中写相对路径/绝对路径/家目录路径。"""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def test_data_processing_paths_exist() -> None:
    """确保数据准备所需目录存在。"""
    cfg = _load_yaml(CONFIG_ROOT / "data_processing.yaml")
    required_dirs = ["raw_data_dir", "interim_dir", "processed_dir", "artifacts_dir", "logs_dir"]

    for key in required_dirs:
        assert key in cfg, f"配置缺失字段：{key}"
        path_value = _resolve(cfg[key])
        assert path_value.exists(), f"{key} 指向的目录不存在：{path_value}"
        assert path_value.is_dir(), f"{key} 应为目录：{path_value}"

    scaler_key = "scaler_output_path"
    assert scaler_key in cfg, f"配置缺失字段：{scaler_key}"
    scaler_path = _resolve(cfg[scaler_key])
    assert scaler_path.parent.exists(), f"{scaler_key} 的父级目录不存在：{scaler_path.parent}"


def test_model_training_output_dirs_exist() -> None:
    """确保模型训练产物目录存在。"""
    cfg = _load_yaml(CONFIG_ROOT / "model_training.yaml")
    output_cfg = cfg.get("output", {})
    expected_dirs = ["models_dir", "artifacts_dir", "figures_dir", "tables_dir", "logs_dir"]

    for key in expected_dirs:
        assert key in output_cfg, f"模型配置缺失输出字段：{key}"
        path_value = _resolve(output_cfg[key])
        # 只要求父级目录存在，因为模型文件往往是新生成的
        assert path_value.parent.exists(), f"{key} 的父级目录不存在：{path_value.parent}"


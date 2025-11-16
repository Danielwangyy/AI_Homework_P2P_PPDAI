"""统一的数据准备与模型训练命令行入口。

该模块仅负责解析命令行参数与调度两条流水线，逻辑轻量，
适合初学者从这里入手了解项目的执行流程。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from ..pipelines.prepare_data import run_pipeline as run_data_pipeline
from ..pipelines.train_models import run_pipeline as run_training_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    之所以单独封装函数，是希望在单元测试或 Notebook 场景下
    也能重用相同的解析逻辑（传入自定义 argv 即可）。
    """
    parser = argparse.ArgumentParser(description="运行完整的数据准备 + 模型训练流程")
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "configs" / "data_processing.yaml"),
        help="数据准备配置文件路径",
    )
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model_training.yaml"),
        help="模型训练配置文件路径",
    )
    parser.add_argument("--skip-data", action="store_true", help="跳过数据准备阶段")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练阶段")
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_config(path_str: str) -> Path:
    """支持相对路径、绝对路径以及 ~ 家目录写法。

    初学者在不同操作系统/工作目录下运行时，常常会遇到路径问题，
    因此这里做一次统一转换，保证最终拿到的是绝对路径。
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def main(argv: Optional[Iterable[str]] = None) -> None:
    """命令行入口。

    步骤说明：
    1. 解析参数；
    2. 解析配置文件路径；
    3. 视参数决定是否执行数据准备/模型训练；
    4. 输出简单的提示信息，便于在终端观察到进度。
    """
    args = _parse_args(argv)
    if args.skip_data and args.skip_train:
        raise SystemExit("不能同时跳过数据准备和模型训练")

    data_config_path = _resolve_config(args.data_config)
    model_config_path = _resolve_config(args.model_config)

    if not args.skip_data:
        print("[ai_homework.cli] 开始执行数据准备流程")
        run_data_pipeline(data_config_path)
        print("[ai_homework.cli] 数据准备流程完成")

    if not args.skip_train:
        print("[ai_homework.cli] 开始执行模型训练流程")
        run_training_pipeline(model_config_path)
        print("[ai_homework.cli] 模型训练流程完成")


if __name__ == "__main__":
    main()


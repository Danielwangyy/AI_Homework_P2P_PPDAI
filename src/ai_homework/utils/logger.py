"""项目统一日志配置。

默认使用 `loguru`，因为它提供了简单的 API 与友好的格式化能力。
若初学者想切换到标准库 `logging`，可以参考此处的初始化思路。
"""
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: Path, name: str = "data_pipeline") -> None:
    """初始化 loguru 日志设置。

    Args:
        log_dir: 日志输出目录。
        name: 日志文件名前缀。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    log_file = log_dir / f"{name}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
    )


def get_logger():
    """返回全局 logger 实例，供其他模块直接使用。"""
    return logger


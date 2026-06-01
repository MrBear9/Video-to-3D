
from pathlib import Path
from loguru import logger
import sys


def _app_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _has_console_stream(stream):
    return stream is not None and hasattr(stream, "write")


def setup_logger():
    logger.remove()
    log_dir = _app_base_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )
    if _has_console_stream(sys.stdout):
        logger.add(sys.stdout, format=log_format, level="INFO")
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )
    return logger


app_logger = setup_logger()

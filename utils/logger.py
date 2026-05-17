
from loguru import logger
import sys


def setup_logger():
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    logger.add("logs/app_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="7 days", level="DEBUG")
    return logger


app_logger = setup_logger()
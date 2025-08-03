import os

from loguru import logger

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 移除默认的控制台输出
logger.remove()


def emoji_format(record):
    level = record["level"].name
    emoji_map = {
        "DEBUG": "🐞",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }
    emoji = emoji_map.get(level, "")
    return (
        f"<green>{record['time']:YYYY-MM-DD HH:mm:ss}</green> | "
        f"{emoji} <level>{level:<8}</level> | "
        f"<cyan>{record['name']}</cyan>:<cyan>{record['function']}</cyan>:<cyan>{record['line']}</cyan> - "
        f"<level>{record['message']}</level>\n"
    )


logger.add(
    "logs/{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    compression="zip",
    format=emoji_format,
    encoding="utf-8",
    level="INFO",
)

# 控制台输出也加 emoji
logger.add(sink=lambda msg: print(msg, end=""), level="INFO", format=emoji_format)

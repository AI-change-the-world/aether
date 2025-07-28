import os

from loguru import logger

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 移除默认的控制台输出
logger.remove()

# 添加文件日志（按天轮转）
logger.add(
    "logs/{time:YYYY-MM-DD}.log",  # 每天一个文件
    rotation="00:00",  # 每天 0 点新文件
    retention="7 days",  # 日志保留 7 天（可选）
    compression="zip",  # 超过保留天数的日志压缩（可选）
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    encoding="utf-8",
    level="INFO",
)

# 你也可以添加一个控制台输出
logger.add(
    sink=lambda msg: print(msg, end=""),  # 控制台输出
    level="INFO",
    format="<blue>{time:HH:mm:ss}</blue> | <level>{level}</level> | <level>{message}</level>",
)

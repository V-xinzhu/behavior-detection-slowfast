import logging
import os

def get_logger(name: str = "default", filename: str = None, level=logging.DEBUG): # type: ignore
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not logger.handlers:
        # 控制台日志
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件日志（如果提供文件名）
        if filename:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(f"logs/{filename}.log", encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

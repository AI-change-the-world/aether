from aether.common.logger import logger


def test_logger():
    logger.info("hello world")
    logger.error("hello world")
    logger.warning("hello world")
    logger.debug("hello world")
    logger.critical("hello world")
    logger.exception("hello world")
    import os
    from datetime import date

    d = date.today()
    assert os.path.exists(f"logs/{d}.log")

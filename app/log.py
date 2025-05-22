import logging

def setup_loggers():
    # === Request Logger: logs only to file ===
    request_logger = logging.getLogger("request_logger")
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
    request_logger.handlers.clear()

    request_file_handler = logging.FileHandler("request.log")
    request_file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    request_logger.addHandler(request_file_handler)

    # === Dev Logger: logs to both console and file ===
    dev_logger = logging.getLogger("dev_logger")
    dev_logger.setLevel(logging.DEBUG)
    dev_logger.propagate = False
    dev_logger.handlers.clear()

    dev_file_handler = logging.FileHandler("dev.log")
    dev_file_handler.setFormatter(logging.Formatter(
        "[DEV] %(asctime)s - %(levelname)s - %(message)s"
    ))

    dev_console_handler = logging.StreamHandler()
    dev_console_handler.setFormatter(logging.Formatter(
        "[DEV] %(asctime)s - %(levelname)s - %(message)s"
    ))

    dev_logger.addHandler(dev_file_handler)
    dev_logger.addHandler(dev_console_handler)

    return request_logger, dev_logger

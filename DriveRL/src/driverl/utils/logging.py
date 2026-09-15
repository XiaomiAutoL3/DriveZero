import logging
import os
import traceback

__all__ = [
    "logger",
    "get_logger",
    "set_level",
    "get_error_log",
]


def set_level(_logger, log_level):
    _logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(name, log_level="INFO"):
    _logger = logging.getLogger(name)
    _logger.propagate = False

    if len(_logger.handlers) > 0:
        return _logger

    log_handler = logging.StreamHandler()  # log to std.err

    base_fmt = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)3s]: %(message)s"
    use_color = (
        os.getenv("DRIVERL_LOG_COLOR", "1") != "0"
        and getattr(log_handler.stream, "isatty", lambda: False)()
    )

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[31m",
        }
        RESET = "\033[0m"

        def __init__(self, fmt: str, enable_color: bool):
            super().__init__(fmt)
            self._enable_color = enable_color

        def format(self, record: logging.LogRecord) -> str:
            msg = super().format(record)
            if self._enable_color and record.levelno in self.COLORS:
                return f"{self.COLORS[record.levelno]}{msg}{self.RESET}"
            return msg

    log_handler.setFormatter(ColorFormatter(base_fmt, use_color))
    _logger.addHandler(log_handler)
    set_level(_logger, log_level)
    return _logger


def get_error_log(error):
    assert isinstance(error, Exception)
    return "".join(traceback.format_exception(None, error, error.__traceback__))


logger = get_logger("DriveRL")

import logging
import sys
from datetime import datetime
from colorama import Fore, Style

# Custom log levels
LOSS_LEVEL = 25  # INFO is 20, WARNING is 30, so 25 is in between.
METRIC_LEVEL = 26  # Evaluation index, higher than LOSS level

logging.addLevelName(LOSS_LEVEL, "LOSS")
logging.addLevelName(METRIC_LEVEL, "METRIC")

class ColoredFormatter(logging.Formatter):
    """Customize the log formatter to make the time and log level colored while keeping the log content white"""

    LEVEL_COLORS = {
        "DEBUG": Fore.MAGENTA,     # Purple (debug information)
        "INFO": Fore.BLUE,         # Blue (normal information)
        "WARNING": Fore.YELLOW,    # Yellow (Warning)
        "ERROR": Fore.RED,         # Red (error)
        "CRITICAL": Fore.LIGHTRED_EX,       # Bright red (critical error)
        "LOSS": Fore.LIGHTCYAN_EX,          # Green (LOSS exclusive color)
        "METRIC": Fore.LIGHTMAGENTA_EX,     # Pink (METRIC evaluation indicator)
    }

    def format(self, record):
        # Colorize time (cyan)
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colored_time = f"{Fore.CYAN}[{log_time}]{Style.RESET_ALL}"

        # Get the log level color
        level_color = self.LEVEL_COLORS.get(record.levelname, Style.RESET_ALL)
        level_name = f"{level_color}{record.levelname}{Style.RESET_ALL}"  # Change the color of INFO / WARNING / ERROR

        # Keep the log content in the default color (white)
        log_message = super().format(record)
        log_message = log_message.replace(record.levelname, level_name)  # Change the color of the log level

        return f"{colored_time} {log_message}"  # Time & level are colored, content is white

class CustomLogger(logging.Logger):
    """Extend Logger and add loss_print and metric_print methods"""

    def loss_print(self, message, *args, **kwargs):
        """Customize LOSS level logs"""
        if self.isEnabledFor(LOSS_LEVEL):
            self._log(LOSS_LEVEL, message, args, **kwargs)

    def metric_print(self, message, *args, **kwargs):
        """Custom METRIC level logs (pink)"""
        if self.isEnabledFor(METRIC_LEVEL):
            self._log(METRIC_LEVEL, message, args, **kwargs)

def get_logger(name="MyLogger"):
    """Creates and returns a colored Logger"""
    logging.setLoggerClass(CustomLogger)  # Setting up a custom Logger class
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Allow all logging levels

    if not logger.handlers:  
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = ColoredFormatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Example usage
if __name__ == '__main__':
    logger = get_logger()
    logger.debug("This is debug information (DEBUG)")
    logger.info("System startup (INFO)")
    logger.warning("Warning: Low disk space (WARNING)")
    logger.error("Error: Unable to connect to database (ERROR)")
    logger.critical("Critical Error: System Crash! (CRITICAL)")

    logger.loss_print("now loss: 0.05")
    logger.loss_print("now loss: 0.8")
    logger.loss_print("now loss: 1.5")

    logger.metric_print("PSNR = 35.6, SSIM = 0.92") 
    logger.metric_print("FID = 12.34, LPIPS = 0.05")

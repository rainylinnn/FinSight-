# 导入日志库
import logging
# 导入路径操作库
import os
# 导入标准错误输出，用于日志文件不可用时的降级提示
import sys
# 导入轮转处理器：日志文件按大小滚动，避免无限增长
from logging.handlers import RotatingFileHandler
# 导入配置类
from base.config import Config

# 单个日志文件上限 10 MB，保留 5 份历史（app.log / app.log.1 ... app.log.4）
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def setup_logging(log_file=Config().LOG_FILE):
    # 创建日志目录（用绝对路径，log_file 只写文件名时也不会出错）
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    # 获取日志器
    logger = logging.getLogger("EduRAG")
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        # 设置控制台处理器级别
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(name)s - %(levelname)s - %(message)s')
        # 为控制台处理器设置格式
        console_handler.setFormatter(formatter)
        # 添加控制台处理器（无论如何都先保证有控制台输出）
        logger.addHandler(console_handler)
        # 创建文件处理器：日志文件被占用/无权限时只降级，不影响程序启动
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8')
            # 设置文件处理器级别
            file_handler.setLevel(logging.INFO)
            # 为文件处理器设置格式
            file_handler.setFormatter(formatter)
            # 添加文件处理器
            logger.addHandler(file_handler)
        except OSError as e:
            print(f"[logger] 日志文件不可写({log_file}): {e}；本次仅输出到控制台", file=sys.stderr)
    # 返回日志器
    return logger


# 初始化日志器
logger = setup_logging()


def process_data(data):
    logger.debug(f"开始处理数据: {data}")
    if not data:
        logger.error("数据为空，无法处理")
        return None
    logger.info("数据处理完成")
    return data.upper()


def main():
    logger.info("程序启动")
    result = process_data("hello")
    if result:
        logger.info(f"处理结果: {result}")
    else:
        logger.warning("处理失败")
    logger.info("程序结束")


if __name__ == "__main__":
    main()

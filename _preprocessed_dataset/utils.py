"""
data/utils.py
辅助函数，供 data_preprocess_v2.py 及同目录其他脚本调用。
"""

import base64
import io
import logging
import os

import PIL.Image as Image


# ─────────────────────────────────────────────────────────────────────────────
# encode_image
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image, format: str = "JPEG", quality: int = 85) -> str:
    """
    将 PIL.Image 对象编码为 base64 字符串（用于 OpenAI vision API）。

    参数：
        image   : PIL.Image 对象，或已是 base64 str（直接返回）
        format  : 保存格式，默认 JPEG（体积小，适合 API 传输）
        quality : JPEG 质量，默认 85

    返回：
        base64 编码的字符串（不含 data URI 前缀）
    """
    if isinstance(image, str):
        # 已经是 base64 字符串，直接返回
        return image

    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if not isinstance(image, Image.Image):
        raise TypeError(f"encode_image 期望 PIL.Image，收到 {type(image)}")

    # JPEG 不支持 RGBA，先转 RGB
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# setup_logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: str = "tmp.log", level: int = logging.INFO) -> logging.Logger:
    """
    创建一个同时输出到文件和控制台的 logger。

    参数：
        name     : logger 名称（通常传 __name__）
        log_file : 日志文件路径，默认 tmp.log（相对于当前工作目录）
        level    : 日志级别，默认 INFO

    返回：
        配置好的 logging.Logger 对象
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（multiprocessing 场景下父进程可能多次调用）
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 文件 handler ──
    os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # ── 控制台 handler ──
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)   # 控制台只显示 WARNING 及以上，避免刷屏
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

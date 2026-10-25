"""Shared helpers for the data-preparation pipeline."""

import base64
import io
import logging
import os

import PIL.Image as Image


# ─────────────────────────────────────────────────────────────────────────────
# encode_image
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image, format: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image as base64 for an OpenAI-compatible vision request.

    Args:
        image: A PIL image or an existing base64 string, returned unchanged.
        format: Image encoding format; defaults to JPEG.
        quality: JPEG quality; defaults to 85.

    Returns:
        The base64 payload without a data-URI prefix."""
    if isinstance(image, str):
        # Return an already encoded base64 string unchanged.
        return image

    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if not isinstance(image, Image.Image):
        raise TypeError(f"encode_image expected a PIL image, received {type(image)}")

    # Convert RGBA to RGB because JPEG does not support an alpha channel.
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# setup_logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: str = "tmp.log", level: int = logging.INFO) -> logging.Logger:
    """Create a logger with file and console handlers.

    Args:
        name: Logger name, usually the calling module name.
        log_file: Log path; defaults to a local temporary log.
        level: Logging level, defaulting to INFO.

    Returns:
        The configured logging.Logger instance."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when initialized repeatedly by multiprocessing workers.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler.
    os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # Restrict console output to warnings and errors.
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

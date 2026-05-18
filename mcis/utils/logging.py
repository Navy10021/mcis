from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_row_count(
    logger: logging.Logger,
    stage: str,
    before: int,
    after: int,
) -> None:
    diff = after - before
    pct = (diff / before * 100) if before > 0 else 0.0
    logger.info(
        "Stage [%s]: %d rows -> %d rows (delta %+d, %+.2f%%)",
        stage, before, after, diff, pct,
    )


def log_output_path(logger: logging.Logger, path: str | Path) -> None:
    logger.info("Wrote: %s", path)

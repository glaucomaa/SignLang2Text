from __future__ import annotations
import logging, sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(save_dir: Path | str | None = None, level="INFO") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(console)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(save_dir) / "train.log")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)

    return logger

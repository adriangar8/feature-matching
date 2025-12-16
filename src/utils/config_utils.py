"""
Configuration utility functions.
"""

from typing import Dict


def get_float(config: Dict, key: str, default: float) -> float:
    val = config.get(key, default)
    return float(val) if val is not None else default


def get_int(config: Dict, key: str, default: int) -> int:
    val = config.get(key, default)
    return int(float(val)) if val is not None else default


def get_bool(config: Dict, key: str, default: bool) -> bool:
    val = config.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('true', '1', 'yes')
    return bool(val) if val is not None else default

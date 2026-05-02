from __future__ import annotations

import collections.abc
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import INPUT_DIR


@dataclass(frozen=True)
class Interval:
    short: str
    start_year: int
    end_year: int
    stats_key: str

    @property
    def display_label(self) -> str:
        return f"{self.start_year}\u2013{self.end_year}"

    @property
    def filename_label(self) -> str:
        return f"{self.start_year}-{self.end_year}"

    @property
    def geometry_col(self) -> str:
        return f"geometry_{self.short}"

    @property
    def surging_col(self) -> str:
        return f"surging_{self.short}"

    @property
    def area_col(self) -> str:
        return f"area_{self.short}"

    @property
    def modified_col(self) -> str:
        return f"modified_{self.short}"


_INTERVALS = (
    Interval("13_18", 2013, 2018, "start_mid"),
    Interval("19_24", 2019, 2024, "mid_end"),
    Interval("13_24", 2013, 2024, "start_end"),
)


def iter_intervals() -> tuple[Interval, ...]:
    return _INTERVALS


def get_interval(short: str) -> Interval:
    for interval in _INTERVALS:
        if interval.short == short:
            return interval
    raise KeyError(short)


def interval_label(short: str) -> str:
    return get_interval(short).display_label


def key_to_interval(key: str) -> str:
    return "_".join(key.split("_")[-2:])


def pandas_str_to_interval(istr: str) -> float | pd.Interval:
    if isinstance(istr, float):
        return np.nan
    c_left = istr[0] == "["
    c_right = istr[-1] == "]"
    closed = {(True, False): "left", (False, True): "right", (True, True): "both", (False, False): "neither"}[c_left, c_right]
    left, right = map(float, istr[1:-1].split(","))
    try:
        return pd.Interval(left, right, closed)
    except Exception:
        return np.nan


def read_aux_bytes(name: str) -> bytes:
    with zipfile.ZipFile(INPUT_DIR / "aux_files.zip") as zip_file:
        return zip_file.read(name)

def read_aux_text(name: str) -> str:
    return read_aux_bytes(name).decode()


def read_aux_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_aux_bytes(name)), **kwargs)

def read_aux_zip(name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(read_aux_bytes(name)))

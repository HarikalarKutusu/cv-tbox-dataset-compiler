"""cv-tbox dataset analyzer - Simple library for helper functions"""

# Standard Lib
import os
import sys
import csv
from typing import Any

# External dependencies
import pandas as pd

# Module
import const as c
import config as conf


def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
        dtype={"ver": str},
    )
    return df


def df_write(df: pd.DataFrame, fpath: Any, mode: Any = "w") -> bool:
    """
    Writes out a dataframe to a file.
    """

    _head: bool = False if mode == "a" else True
    # Create/override the file
    df.to_csv(
        fpath,
        mode=mode,
        header=_head,
        index=False,
        encoding="utf-8",
        sep="\t",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    # float_format="%.4f"
    if conf.VERBOSE:
        print(f"Generated: {fpath} Records={df.shape[0]}")
    return True


def df_int_convert(x: pd.Series) -> Any:
    """Convert columns to int if possible"""
    try:
        return x.astype(int)
    except:
        return x


def list2str(lst: list[Any]) -> str:
    """Convert a list into a string"""
    return c.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: list[list[Any]]) -> str:
    """Convert an array (list of lists) into a string"""
    return c.SEP_ROW.join(list2str(x) for x in arr)


# Calc CV_DIR - Different for v1-4 !!!
def calc_cv_dir_name(cv_idx: int, cv_ver: str) -> str:
    """Create CV dataset main directory name"""
    if cv_ver in ["1", "2", "3", "4"]:
        return "cv-corpus-" + cv_ver
    else:
        return "cv-corpus-" + cv_ver + "-" + c.CV_DATES[cv_idx]


def dec3(x: float) -> float:
    """Make to 3 decimals"""
    return round(1000 * x) / 1000

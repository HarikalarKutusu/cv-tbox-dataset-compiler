"""cv-tbox dataset analyzer - Simple library for helper functions"""

# Standard Lib
import os
import sys
import csv
import json
from typing import Literal, Any
from urllib.request import urlopen

# External dependencies
import pandas as pd

# Module
import const as c
import conf

#
# Init
#


def init_directories(basedir: str) -> None:
    """Creates data directory structures"""
    all_locales: list[str] = get_locales(c.CV_VERSIONS[-1])
    data_dir: str = os.path.join(basedir, c.DATA_DIRNAME)
    for lc in all_locales:
        os.makedirs(os.path.join(data_dir, c.CD_DIRNAME, lc), exist_ok=True)
        os.makedirs(os.path.join(data_dir, c.RES_DIRNAME, c.TSV_DIRNAME, lc), exist_ok=True)
        os.makedirs(os.path.join(data_dir, c.RES_DIRNAME, c.JSON_DIRNAME, lc), exist_ok=True)
    for ver in c.CV_VERSIONS:
        ver_lc: list[str] = get_locales(ver)
        for lc in ver_lc:
            ds_dir: str = os.path.join(calc_dataset_prefix(ver), lc)
            os.makedirs(
                os.path.join(data_dir, c.TC_DIRNAME, ds_dir),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(data_dir, c.VC_DIRNAME, ds_dir),
                exist_ok=True,
            )


#
# DataFrames
#


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
    except ValueError as e: # pylint: disable=W0612
        return x


#
# Common Voice Dataset & API Related
#


def is_version_valid(ver: str) -> Literal[True]:
    """Check a ver string in valid"""

    if not ver in c.CV_VERSIONS:
        print(f"FATAL: {ver} is not a supported Common Voice version.")

        sys.exit(1)
    return True


def calc_dataset_prefix(ver: str) -> str:
    """Build the dataset string from version (valid for > v4)"""

    if is_version_valid(ver):
        inx: int = c.CV_VERSIONS.index(ver)
        if ver in ["1", "2", "3", "4"]:
            return f"cv-corpus-{ver}"
        return f"cv-corpus-{ver}-{c.CV_DATES[inx]}"
    return ""


def get_from_cv_api(url: str) -> Any:
    """Get data from CV API"""
    try:
        res: Any = urlopen(url)
    except RuntimeError as e:
        print(f"Metadata at {url} could not be located!")
        print(f"Error: {e}")
        sys.exit(-1)
    return json.loads(res.read())


def get_locales(ver: str) -> list[str]:
    """Get data from API 'datasets' endpoint"""
    jdict: Any = get_from_cv_api(
        f"{c.CV_DATASET_BASE_URL}/{calc_dataset_prefix(ver)}.json"
    )
    jlocales: Any = jdict["locales"]
    locales: list[str] = []
    for loc, _data in jlocales.items():
        locales.append(loc)
    locales.sort()
    return locales


#
# Conversion
#


def list2str(lst: list[Any]) -> str:
    """Convert a list into a string"""
    return c.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: list[list[Any]]) -> str:
    """Convert an array (list of lists) into a string"""
    return c.SEP_ROW.join(list2str(x) for x in arr)


#
# Numbers
#


def dec3(x: float) -> float:
    """Make to 3 decimals"""
    return round(1000 * x) / 1000

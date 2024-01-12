""" Get all CV locales with released datasets from API """

# Standard Lib
import sys
import json
from typing import Literal, Any
from urllib.request import urlopen

# External dependencies

# Module
from const import CV_VERSIONS, CV_DATES, CV_DATASET_BASE_URL


def is_version_valid(ver: str) -> Literal[True]:
    """Check a ver string in valid"""

    if not ver in CV_VERSIONS:
        print(f"FATAL: {ver} is not a supported Common Voice version.")

        sys.exit(1)
    return True


def calc_dataset_prefix(ver: str) -> str:
    """Build the dataset string from version (valid for > v4)"""

    if is_version_valid(ver):
        inx: int = CV_VERSIONS.index(ver)
        if ver in ["1", "3", "4"]:
            return f"cv-corpus-{ver}"
        return f"cv-corpus-{ver}-{CV_DATES[inx]}"
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
    jdict: Any = get_from_cv_api(f"{CV_DATASET_BASE_URL}/{calc_dataset_prefix(ver)}.json")
    jlocales: Any = jdict["locales"]
    locales: list[str] = []
    for loc, _data in jlocales.items():
        locales.append(loc)
    locales.sort()
    return locales

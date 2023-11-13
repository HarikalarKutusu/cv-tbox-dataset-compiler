""" Get all CV locales with released datasets from API """

# Standard Lib
import sys
import json
from typing import Literal, Any
from urllib.request import urlopen

# External dependencies

# Module
from const import CV_VERSIONS, CV_DATES, CV_DATASET_BASE_URL


def is_version_valid(ver) -> Literal[True]:
    """Check a ver string in valid"""

    if not ver in CV_VERSIONS:
        print(f"FATAL: {ver} is not a supported Common Voice version.")

        sys.exit(1)
    return True


def calc_dataset_prefix(ver) -> str | None:
    """Build the dataset string from version (valid for > v4)"""

    if is_version_valid(ver):
        inx: int = CV_VERSIONS.index(ver)
        return f"cv-corpus-{ver}-{CV_DATES[inx]}"
    return None


def get_locales(ver: str) -> list[str]:
    """Get data from API 'datasets' endpoint"""
    url: str = f"{CV_DATASET_BASE_URL}/{calc_dataset_prefix(ver)}.json"
    try:
        res: Any = urlopen(url)
    except:
        print(f"Metadata for version {ver} could not be located!")
        sys.exit()
    jdict: Any = json.loads(res.read())
    jlocales: Any = jdict["locales"]
    locales: list[str] = []
    for loc, _data in jlocales.items():
        locales.append(loc)
    locales.sort()
    return locales

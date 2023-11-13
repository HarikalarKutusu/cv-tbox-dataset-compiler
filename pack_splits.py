#!/usr/bin/env python3
"""
###########################################################################
# pack_splits.py
#
# Pack split files into .xz for download from cv-tbox-dataset-analyzer
#
# Use:
# python pack_splits.py
#
#
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################
"""

import os
import sys
import shutil
import glob
from datetime import datetime, timedelta
import multiprocessing as mp

import psutil

# This package
import const
from get_locales import get_locales

import config as conf

#
# Constants
#

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Program parameters
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
ALL_LOCALES: list[str] = get_locales(const.CV_VERSIONS[-1])


def handle_ds(dspath: str) -> None:
    """Handle a single version/lc in multi-processing"""

    plist: list[str] = dspath.split(os.sep)
    corpus: str = plist[-2]
    ver: str = corpus.split("-")[2]
    lc: str = plist[-1]
    dest_dir: str = os.path.join(HERE, "data", "results", "json", lc)
    print(f"Compressing Dataset Splits for {corpus} - {lc}")
    for algo in const.ALGORITHMS:
        if os.path.isdir(os.path.join(dspath, algo)):  # check if algo exists at source
            tarpath: str = os.path.join(dest_dir, f"{lc}_{ver}_{algo}")
            # Skip existing?
            if not os.path.isfile(tarpath + ".tar.xz") or conf.FORCE_CREATE_COMPRESSED:
                shutil.make_archive(
                    base_name=tarpath, format="xztar", root_dir=dspath, base_dir=algo
                )


# MAIN PROCESS
def main() -> None:
    """Main loop to compress split files for download"""

    print("=== Split Compressor for cv-tbox-dataset-analyzer ===")
    start_time: datetime = datetime.now()

    tc_base_dir: str = os.path.join(HERE, "data", "voice-corpus")

    # Get a list of available language codes in every version
    dspaths: list[str] = glob.glob(
        os.path.join(tc_base_dir, "**", const.ALGORITHMS[0]), recursive=True
    )
    for inx, dspath in enumerate(dspaths):
        dspaths[inx] = os.path.split(dspath)[0]  # get rid of the final part

    # extra line is for progress line
    print(f"Compressing for {len(dspaths)} datasets...\n")

    with mp.Pool(psutil.cpu_count(logical=False)) as pool:
        pool.map(handle_ds, dspaths)

    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f"Finished compressing for {len(dspaths)} datasets in {str(process_timedelta)}"
        + f" avg={process_seconds/len(dspaths)} sec/locale"
    )


if __name__ == "__main__":
    main()

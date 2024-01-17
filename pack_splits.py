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
import const as c
import conf
from lib import get_locales


#
# Constants
#

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Program parameters
PROC_COUNT: int = int(1.5 * psutil.cpu_count(logical=True))  # OVER usage
BATCH_SIZE: int = 5
ALL_LOCALES: list[str] = get_locales(c.CV_VERSIONS[-1])


def handle_ds(dspath: str) -> None:
    """Handle a single version/lc in multi-processing"""

    plist: list[str] = dspath.split(os.sep)
    corpus: str = plist[-2]
    ver: str = corpus.split("-")[2]
    lc: str = plist[-1]
    upload_dir: str = os.path.join(
        conf.COMPRESSED_RESULTS_BASE_DIR, c.UPLOAD_DIRNAME, lc
    )
    uploaded_dir: str = os.path.join(
        conf.COMPRESSED_RESULTS_BASE_DIR, c.UPLOADED_DIRNAME, lc
    )
    os.makedirs(upload_dir, exist_ok=True)
    print(f"Compressing Dataset Splits for {corpus} - {lc}", flush=True)
    for algo in c.ALGORITHMS:
        if os.path.isdir(os.path.join(dspath, algo)):  # check if algo exists at source
            tarpath1: str = os.path.join(upload_dir, f"{lc}_{ver}_{algo}")
            tarpath2: str = os.path.join(uploaded_dir, f"{lc}_{ver}_{algo}")
            # Skip existing?
            if (
                not os.path.isfile(tarpath2 + ".tar.xz")
                and not os.path.isfile(tarpath1 + ".tar.xz")
            ) or conf.FORCE_CREATE_COMPRESSED:
                shutil.make_archive(
                    base_name=tarpath1, format="xztar", root_dir=dspath, base_dir=algo
                )


# MAIN PROCESS
def main() -> None:
    """Main loop to compress split files for download"""

    print("=== Split Compressor for cv-tbox-dataset-analyzer ===")
    start_time: datetime = datetime.now()

    # Get a list of available language codes in every version
    dspaths: list[str] = glob.glob(
        os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME, "**", c.ALGORITHMS[0]),
        recursive=True,
    )
    for inx, dspath in enumerate(dspaths):
        dspaths[inx] = os.path.split(dspath)[0]  # get rid of the final part

    # extra line is for progress line
    print(f"Compressing for {len(dspaths)} datasets...\n")

    with mp.Pool(processes=PROC_COUNT, maxtasksperchild=BATCH_SIZE) as pool:
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

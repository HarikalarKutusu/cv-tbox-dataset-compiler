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

from collections import Counter
import os
import sys
import shutil
import glob
import multiprocessing as mp

import psutil
from tqdm import tqdm

# This package
import const as c
import conf
from lib import init_directories, report_results
from typedef import Globals


#
# Constants
#

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Program parameters
PROC_COUNT: int = min(60, int(2 * psutil.cpu_count(logical=True)))  # OVER usage
BATCH_SIZE: int = 5

g: Globals = Globals()

ver_counter: Counter[str] = Counter()
lc_counter: Counter[str] = Counter()
algo_counter: Counter[str] = Counter()


def handle_ds(p: str) -> str:
    """Handle a single version/lc in multi-processing"""

    plist: list[str] = p.split(os.sep)
    algo: str = plist[-1]
    lc: str = plist[-2]
    ver: str = plist[-3].split("-")[2]

    upload_dir: str = os.path.join(
        conf.COMPRESSED_RESULTS_BASE_DIR, c.UPLOAD_DIRNAME, lc
    )

    os.makedirs(upload_dir, exist_ok=True)
    shutil.make_archive(
        base_name=os.path.join(upload_dir, f"{lc}_{ver}_{algo}"),
        format="xztar",
        root_dir=os.path.split(p)[0],
        base_dir=algo,
    )
    g.processed_algo += 1

    # report back
    return f"{ver}_{lc}_{algo}"


# MAIN PROCESS
def main() -> None:
    """Main loop to compress split files for download"""

    # Get a list of available language codes in every version
    dspaths: list[str] = glob.glob(
        os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME, "**", c.ALGORITHMS[0]),
        recursive=True,
    )
    dspaths = [os.path.split(p)[0] for p in dspaths]

    #
    # Drop already compressed
    #
    compressed_list: list[str] = glob.glob(
        os.path.join(conf.COMPRESSED_RESULTS_BASE_DIR, "**", "*.tar.xz"), recursive=True
    )
    compressed_list = [
        p.split(os.sep)[-1].replace(".tar.xz", "") for p in compressed_list
    ]

    final_list: list[str] = []
    for p in dspaths:
        plist: list[str] = p.split(os.sep)
        ver: str = plist[-2].split("-")[2]
        lc: str = plist[-1]
        ver_counter.update([ver])
        lc_counter.update([lc])
        for algo in c.ALGORITHMS:
            maybe_p: str = os.path.join(p, algo)
            if os.path.isdir(maybe_p):
                algo_counter.update([algo])
                if not "_".join([lc, ver, algo]) in compressed_list:
                    final_list.append(maybe_p)
                else:
                    g.skipped_exists += 1
            else:
                g.skipped_nodata += 1

    total_cnt: int = len(final_list)

    # record totals
    g.total_ver = len(ver_counter.values())
    g.total_lc = lc_counter.total()
    g.total_algo = algo_counter.total()
    g.total_splits = 3 * g.total_algo

    ver_counter.clear()
    lc_counter.clear()
    algo_counter.clear()

    #
    # Process with multi-processing
    #
    if total_cnt > 0:
        print(
            f"Compressing for {len(final_list)} algorithmns in {len(dspaths)} datasets...\n"
        )
        with mp.Pool(processes=PROC_COUNT, maxtasksperchild=BATCH_SIZE) as pool:
            # pool.map(handle_ds, dspaths)
            with tqdm(total=total_cnt, desc="Datasets") as pbar:
                for res in pool.imap_unordered(
                    handle_ds, final_list, chunksize=BATCH_SIZE
                ):
                    if conf.VERBOSE:
                        pbar.write(f"Finished: {res}")
                    reslist: list[str] = res.split("_")
                    ver_counter.update([reslist[0]])
                    lc_counter.update([reslist[1]])
                    algo_counter.update([reslist[2]])
                    pbar.update()

    # done
    g.processed_ver = len(ver_counter.keys())
    g.processed_lc = len(lc_counter.keys())
    g.processed_algo = len(algo_counter.keys())
    print(ver_counter.most_common())
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-analyzer - Split Compressiom ===")
    init_directories(HERE)
    main()

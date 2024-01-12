#!/usr/bin/env python3
"""cv-tbox Dataset Compiler - Voice-Corpus Splits Compilation Phase"""
###########################################################################
# split_compile.py
#
# Tool for internal use, copies files from other directories and creates
# the necessary directory structure for this tool.
#
# Use:
# python split_compile.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
import os
import sys
import shutil
import glob
from datetime import datetime

# External dependencies
from tqdm import tqdm

# Module
import const as c
import conf
from typedef import Globals
from lib import calc_dataset_prefix, dec3

# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

g: Globals = Globals(
    total_ver=len(c.CV_VERSIONS),
    total_algo=len(c.ALGORITHMS),
)


# MAIN PROCESS
def main() -> None:
    """Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler"""

    # Destination voice corpus
    vc_dir_base: str = os.path.join(HERE, "data", "voice-corpus")
    # Destination clip durations
    cd_dir_base: str = os.path.join(HERE, "data", "clip-durations")
    # CV Release directory name
    cv_dir_name: str = ""

    src_dir: str = ""
    dst_dir: str = ""

    #
    # Subs
    #

    def handle_locale(lc: str) -> None:
        """Handles one locale's data"""

        nonlocal src_dir, dst_dir

        g.processed_lc += 1
        g.processed_algo += g.total_algo

        # copy splitting algorithm independent files
        src_dir = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, lc)
        dst_dir = os.path.join(vc_dir_base, cv_dir_name, lc)
        tsv_fpath: str = ""

        if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
            # os.makedirs(os.path.join(dst_dir, c.ALGORITHMS[0]), exist_ok=True)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in c.EXTENDED_BUCKET_FILES:
                tsv_fpath = os.path.join(src_dir, fn)
                if os.path.isfile(tsv_fpath):
                    shutil.copy2(tsv_fpath, dst_dir)
        # else:
        #     g.skipped_exists += 1

        # copy default splits (s1)

        # copy to s1
        dst_dir = os.path.join(vc_dir_base, cv_dir_name, lc, c.ALGORITHMS[0])
        if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for fn in c.SPLIT_FILES:
                tsv_fpath = os.path.join(src_dir, fn)
                if os.path.isfile(tsv_fpath):
                    shutil.copy2(tsv_fpath, dst_dir)
        else:
            g.skipped_exists += 1

        # copy other splitting algorithms' split files

        for algo in c.ALGORITHMS[1:]:
            # check if exists to copy to "algo dir"
            src_dir = os.path.join(conf.SRC_BASE_DIR, algo, cv_dir_name, lc)
            if os.path.isdir(src_dir):
                dst_dir = os.path.join(vc_dir_base, cv_dir_name, lc, algo)
                if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fn in c.SPLIT_FILES:
                        tsv_fpath = os.path.join(src_dir, fn)
                        if os.path.isfile(tsv_fpath):
                            shutil.copy2(tsv_fpath, dst_dir)
                else:
                    g.skipped_exists += 1

        # clip durations table, the one from the latest version (v15.0+) is valid
        # for all CV versions (not taking deletions into account)
        if cv_ver == c.CV_VERSIONS[-1]:
            dst_dir = os.path.join(cd_dir_base, lc)
            os.makedirs(dst_dir, exist_ok=True)
            # With v15.0, we have the provided "clip_durations.tsv" (duration is in ms)
            cd_file: str = os.path.join(
                conf.SRC_BASE_DIR,
                c.ALGORITHMS[0],
                cv_dir_name,
                lc,
                c.CLIP_DURATIONS_FILE,
            )
            if os.path.isfile(cd_file):
                shutil.copy2(cd_file, dst_dir)
            else:
                # [TODO]: If it is not found, we need to create it.
                print(f"WARNING: clip_durations.tsv file not found for {cv_ver} - {lc}")

    #
    # Main
    #

    print(
        "=== Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler ==="
    )

    # Loop all versions
    # pbar_ver = tqdm(c.CV_VERSIONS, desc="Versions", total=g.total_ver, unit=" Version")
    for cv_ver in c.CV_VERSIONS:
        # Check if it exists in source (check "s1", if not there, it is nowhere)
        cv_dir_name = calc_dataset_prefix(cv_ver)
        if not os.path.isdir(
            os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name)
        ):
            continue  # Does not exist, so skip

        # Create destination
        dst_dir = os.path.join(vc_dir_base, cv_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        # Get a  list of available language codes
        lc_paths: list[str] = glob.glob(
            pathname=os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, "*"),
            recursive=False,
        )
        lc_list: list[str] = [os.path.split(p)[-1] for p in lc_paths]
        lc_cnt: int = len(lc_list)

        g.processed_ver += 1
        g.total_lc += lc_cnt

        # print("=" * 80)
        # print(f"Processing {lc_cnt} locales in {cv_dir_name}")

        pbar_lc = tqdm(
            lc_list, desc=("   v" + cv_ver)[-5:], total=lc_cnt, unit=" Locale"
        )
        for lc in lc_list:
            handle_locale(lc)
            pbar_lc.update()

        pbar_lc.close()
        # pbar_ver.update()

    # done
    process_seconds: float = (datetime.now() - g.start_time).total_seconds()
    print("=" * 80)
    print(f"Total\t\t: Ver: {g.total_ver} LC: {g.total_lc} Algo: {g.total_algo}")
    print(f"Scanned\t\t: Ver: {g.processed_ver} LC: {g.processed_lc} Algo: {g.processed_algo}")
    print(f"Skipped\t\t: Algo: {g.skipped_exists}")
    print(f"Duration(s)\t: Total: {dec3(process_seconds)} Avg: {dec3(process_seconds/ (g.processed_algo - g.skipped_exists))}")

if __name__ == "__main__":
    main()

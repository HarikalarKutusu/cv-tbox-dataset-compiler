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
from datetime import datetime, timedelta

# External dependencies

# Module
import const as c
import conf

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)


# MAIN PROCESS
def main() -> None:
    """Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler"""

    overall_cnt: int = 0
    cnt: int = 0
    src_dir: str = ""
    dst_dir: str = ""

    #
    # Subs
    #

    def handle_locale() -> None:
        """Handles one locale's data"""

        nonlocal overall_cnt, cnt, src_dir, dst_dir

        overall_cnt += 1
        cnt += 1

        txt: str = f"Processing version {cv_ver} ({cv_idx+1}/{len(c.CV_VERSIONS)}) locale {cnt}/{len(lc_list)} : {lc}"
        print(txt if conf.VERBOSE else "\033[F" + txt + " " * 20)

        # copy splitting algorithm independent files

        src_dir = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, lc)
        dst_dir = os.path.join(vc_dir_base, cv_dir_name, lc)
        # print(os.path.join(src_dir, fn), " => ", dst_dir)
        tsv_fpath: str = ""
        # print("\n=> ", dst_dir, "\n")

        if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
            # os.makedirs(os.path.join(dst_dir, c.ALGORITHMS[0]), exist_ok=True)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in c.EXTENDED_BUCKET_FILES:
                tsv_fpath = os.path.join(src_dir, fn)
                if os.path.isfile(tsv_fpath):
                    shutil.copy2(tsv_fpath, dst_dir)

        # copy default splits (s1)

        # copy to s1
        dst_dir = os.path.join(vc_dir_base, cv_dir_name, lc, c.ALGORITHMS[0])
        if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for fn in c.SPLIT_FILES:
                tsv_fpath = os.path.join(src_dir, fn)
                if os.path.isfile(tsv_fpath):
                    shutil.copy2(tsv_fpath, dst_dir)

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
                print("\n\nWARNING: $clip_durations.tsv file not found for", lc, "\n")

    #
    # Main
    #

    print(
        "=== Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler ==="
    )

    start_time: datetime = datetime.now()

    # Destination voice corpus
    vc_dir_base: str = os.path.join(HERE, "data", "voice-corpus")
    # Destination clip durations
    cd_dir_base: str = os.path.join(HERE, "data", "clip-durations")
    # CV Release directory name
    cv_dir_name: str = ""

    # Loop all versions
    for cv_idx, cv_ver in enumerate(c.CV_VERSIONS):
        # Calc CV_DIR - Different for v1-4 !!!
        if cv_ver in ["1", "2", "3", "4"]:
            cv_dir_name = "cv-corpus-" + cv_ver
        else:
            cv_dir_name = "cv-corpus-" + cv_ver + "-" + c.CV_DATES[cv_idx]
        # Check if it exists in source (check "s1", if not there, it is nowhere)
        if not os.path.isdir(
            os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name)
        ):
            continue  # Does, not exist, so skip

        print("=" * 80)
        print(f"Processing locales in {cv_dir_name}\n")

        # Create destination
        dst_dir = os.path.join(vc_dir_base, cv_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        # Get a  list of available language codes
        lc_paths: list[str] = glob.glob(
            pathname=os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, "*"),
            recursive=False,
        )
        lc_list: list[str] = []
        for lc_path in lc_paths:
            lc: str = os.path.split(lc_path)[1]
            lc_list.append(lc)

        # ALGO-1 (s1 - default splits)
        for lc in lc_list:
            handle_locale()

    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f"\nFinished copy of {overall_cnt} datasets in {str(process_timedelta)},"
        + f" avg={process_seconds/overall_cnt} sec"
    )


if __name__ == "__main__":
    main()

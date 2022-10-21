#!/usr/bin/env python3

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

import sys
import os
import shutil
import glob
from datetime import datetime, timedelta


HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

import const
import config as conf

# MAIN PROCESS
def main() -> None:
    print('=== Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler ===')

    start_time: datetime = datetime.now()

    vc_dir_base: str = os.path.join(HERE, 'data', "voice-corpus")
    # tc_dir_base: str = os.path.join(HERE, 'data', "text-corpus")
    cd_dir_base: str = os.path.join(HERE, 'data', "clip-durations")

    # Loop all versions
    overall_cnt: int = 0
    for cv_idx, cv_ver in enumerate(const.CV_VERSIONS):
        # Calc CV_DIR - TODO different for v1-4 !!!
        cv_dir_name: str = "cv-corpus-" + cv_ver + "-" + const.CV_DATES[cv_idx]
        # Check if it exists in source (check "s1", if not there, it is nowhere)
        if not os.path.isdir(os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[0], cv_dir_name)):
            continue # Does, not exist, so skip

        print(f'Processing locales in {cv_dir_name}\n')

        # Create destination
        dst_dir: str = os.path.join(vc_dir_base, cv_dir_name)
        os.makedirs(dst_dir, exist_ok=True)
        # Get a  list of available language codes
        lc_paths: "list[str]" = glob.glob(
            os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[0], cv_dir_name, '*'),
            recursive=False)
        lc_list: "list[str]" = []
        for lc_path in lc_paths:
            lc: str = os.path.split(lc_path)[1]
            lc_list.append(lc)

        # ALGO-1 (s1 - default splits)
        cnt: int = 0
        for lc in lc_list:
            overall_cnt += 1
            cnt += 1
            print('\033[F' + ' ' * 80)
            print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')

            # copy splitting algorithm independent files
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[0], cv_dir_name, lc)
            dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc)
            os.makedirs(os.path.join(dst_dir, const.ALGORITHMS[0]), exist_ok=True)
            for fn in ["validated.tsv", "invalidated.tsv", "other.tsv", "reported.tsv"]:
                if os.path.isfile(os.path.join(src_dir, fn)):
                    shutil.copy2(os.path.join(src_dir, fn), dst_dir)

            # copy to s1
            dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, const.ALGORITHMS[0])
            for fn in ["train.tsv", "dev.tsv", "test.tsv"]:
                if os.path.isfile(os.path.join(src_dir, fn)):
                    shutil.copy2(os.path.join(src_dir, fn), dst_dir)

            # check if exists to copy to s99
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[1], cv_dir_name, lc)
            if os.path.isdir(src_dir):
                dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, const.ALGORITHMS[1])
                os.makedirs(dst_dir, exist_ok=True)
                for fn in ["train.tsv", "dev.tsv", "test.tsv"]:
                    if os.path.isfile(os.path.join(src_dir, fn)):
                        shutil.copy2(os.path.join(src_dir, fn), dst_dir)

            # check if exists to copy to v1
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[2], cv_dir_name, lc)
            if os.path.isdir(src_dir):
                dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, const.ALGORITHMS[2])
                os.makedirs(dst_dir, exist_ok=True)
                for fn in ["train.tsv", "dev.tsv", "test.tsv"]:
                    if os.path.isfile(os.path.join(src_dir, fn)):
                        shutil.copy2(os.path.join(src_dir, fn), dst_dir)

            # special case for clip durations, we need to find it
            dst_dir: str = os.path.join(cd_dir_base, lc)
            os.makedirs(dst_dir, exist_ok=True)
            cd_file: str = '$clip_durations.tsv'
            search_paths: list[str] = [
                os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[0], cv_dir_name, lc, 'clips', cd_file),
                os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[1], cv_dir_name, lc, 'clips', cd_file),
                os.path.join(conf.SRC_BASE_DIR, const.ALGORITHMS[2], cv_dir_name, lc, 'clips', cd_file)
            ]
            for p in search_paths:
                if os.path.isfile(p):
                    shutil.copy2(p, dst_dir)
                    continue

    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'\nFinished copy of {overall_cnt} datasets in {str(process_timedelta)}, avg={process_seconds/overall_cnt} sec')


main()

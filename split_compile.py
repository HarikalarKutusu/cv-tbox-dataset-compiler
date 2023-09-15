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

import sys, os, shutil, glob
from datetime import datetime, timedelta

import const as c
import config as conf

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# MAIN PROCESS
def main() -> None:
    print('=== Data Algorithms/Splits Preparation Process for cv-tbox-dataset-compiler ===')

    start_time: datetime = datetime.now()

    # Destination voice corpus
    vc_dir_base: str = os.path.join(HERE, 'data', "voice-corpus")
    # Destination clip durations
    cd_dir_base: str = os.path.join(HERE, 'data', "clip-durations")

    # Loop all versions
    overall_cnt: int = 0
    for cv_idx, cv_ver in enumerate(c.CV_VERSIONS):
        # Calc CV_DIR - Different for v1-4 !!!
        if cv_ver in ['1', '2', '3', '4']:
            cv_dir_name: str = "cv-corpus-" + cv_ver
        else:
            cv_dir_name: str = "cv-corpus-" + cv_ver + "-" + c.CV_DATES[cv_idx]
        # Check if it exists in source (check "s1", if not there, it is nowhere)
        if not os.path.isdir(os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name)):
            continue # Does, not exist, so skip

        print(f'Processing locales in {cv_dir_name}\n')

        # Create destination
        dst_dir: str = os.path.join(vc_dir_base, cv_dir_name)
        os.makedirs(dst_dir, exist_ok=True)
        # Get a  list of available language codes
        lc_paths: "list[str]" = glob.glob(
            os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, '*'),
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
            print(f'\033[FProcessing version {cv_ver} ({cv_idx+1}/{len(c.CV_VERSIONS)}) locale {cnt}/{len(lc_list)} : {lc}')

            # copy splitting algorithm independent files
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, lc)
            dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc)
            # print(os.path.join(src_dir, fn), " => ", dst_dir)
            print("\n=> ", dst_dir, "\n")
            if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
                # os.makedirs(os.path.join(dst_dir, c.ALGORITHMS[0]), exist_ok=True)
                os.makedirs(dst_dir, exist_ok=True)
                for fn in c.EXTENDED_BUCKET_FILES:
                    tsvFile: str = os.path.join(src_dir, fn)
                    if os.path.isfile(tsvFile):
                        shutil.copy2(tsvFile, dst_dir)

            # copy to s1
            dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, c.ALGORITHMS[0])
            if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
                for fn in c.SPLIT_FILES:
                    tsvFile: str = os.path.join(src_dir, fn)
                    if os.path.isfile(tsvFile):
                        shutil.copy2(tsvFile, dst_dir)

            # check if exists to copy to s99
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[1], cv_dir_name, lc)
            if os.path.isdir(src_dir):
                dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, c.ALGORITHMS[1])
                if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fn in c.SPLIT_FILES:
                        tsvFile: str = os.path.join(src_dir, fn)
                        if os.path.isfile(tsvFile):
                            shutil.copy2(tsvFile, dst_dir)

            # check if exists to copy to v1
            src_dir: str = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[2], cv_dir_name, lc)
            if os.path.isdir(src_dir):
                dst_dir: str = os.path.join(vc_dir_base, cv_dir_name, lc, c.ALGORITHMS[2])
                if conf.FORCE_CREATE_SPLIT_STATS or not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fn in c.SPLIT_FILES:
                        tsvFile: str = os.path.join(src_dir, fn)
                        if os.path.isfile(tsvFile):
                            shutil.copy2(tsvFile, dst_dir)

            # clip durations table, the one from the latest version is valid for all CV versions (not taking deletions into account)
            # This is valid for v15.0+
            if cv_ver == c.CV_VERSIONS[-1]:
                dst_dir: str = os.path.join(cd_dir_base, lc)
                os.makedirs(dst_dir, exist_ok=True)
                # With v15.0, we have the provided "clip_durations.tsv" (duration is in ms)
                cd_file: str = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], cv_dir_name, lc, c.CLIP_DURATIONS_FILE)
                if os.path.isfile(cd_file):
                    shutil.copy2(cd_file, dst_dir)
                else:
                    # TODO: If it is not found, we need to create it.
                    print('\n\nWARNING: $clip_durations.tsv file not found for', lc, '\n')


    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'\nFinished copy of {overall_cnt} datasets in {str(process_timedelta)}, avg={process_seconds/overall_cnt} sec')


main()

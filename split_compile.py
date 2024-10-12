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

# External dependencies
from tqdm import tqdm

# Module
import const as c
import conf
from typedef import Globals
from lib import calc_dataset_prefix, init_directories, report_results

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
    vc_dir_base: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME)
    # Destination clip durations
    cd_dir_base: str = os.path.join(conf.DATA_BASE_DIR, c.CD_DIRNAME)
    # CV Release directory name
    ver_dir: str = ""

    src_dir: str = ""
    dst_dir: str = ""

    #
    # Subs
    #
    def copy_file_list(src_dir: str, dst_dir: str, files: list[str]) -> None:
        """Copy a list of files from source to destination"""
        for fn in files:
            src_fpath: str = os.path.join(src_dir, fn)
            if os.path.isfile(src_fpath):
                shutil.copy2(src_fpath, dst_dir)

    def handle_locale(lc: str) -> None:
        """Handles one locale's data"""

        nonlocal src_dir, dst_dir

        g.processed_lc += 1
        g.processed_algo += g.total_algo

        # copy splitting algorithm independent files
        src_dir = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], ver_dir, lc)
        dst_dir = os.path.join(vc_dir_base, ver_dir, lc)
        src_tsv_fpath: str

        files_to_copy: list[str] = c.EXTENDED_BUCKET_FILES.copy()
        files_to_copy.extend(c.TC_BUCKET_FILES)

        if conf.FORCE_CREATE_VC_STATS or not os.path.isfile(
            os.path.join(dst_dir, "validated.tsv")
        ):
            copy_file_list(src_dir, dst_dir, c.EXTENDED_BUCKET_FILES)
            copy_file_list(src_dir, dst_dir, c.TC_BUCKET_FILES)

        # copy all splitting algorithms' split files - check existance thou
        for algo in c.ALGORITHMS:
            # check if exists to copy to "algo dir"
            src_dir = os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], ver_dir, lc)
            if not os.path.isfile(os.path.join(src_dir, "train.tsv")):
                # no such split at source
                g.skipped_nodata += 1
                continue
            dst_dir = os.path.join(vc_dir_base, ver_dir, lc, algo)
            if (
                os.path.isfile(os.path.join(dst_dir, "train.tsv"))
                and not conf.FORCE_CREATE_VC_STATS
            ):
                # destination already exists
                g.skipped_exists += 1
                continue
            # now we can copy
            os.makedirs(dst_dir, exist_ok=True)
            copy_file_list(src_dir, dst_dir, c.SPLIT_FILES)

        # clip durations table, the one from the latest version (v15.0+) is valid
        # for all CV versions (not taking deletions into account)
        if ver == c.CV_VERSIONS[-1]:
            dst_dir = os.path.join(cd_dir_base, lc)
            # With v15.0, we have the provided "clip_durations.tsv" (duration is in ms)
            cd_file: str = os.path.join(
                conf.SRC_BASE_DIR,
                c.ALGORITHMS[0],
                ver_dir,
                lc,
                c.CLIP_DURATIONS_FILE,
            )
            if os.path.isfile(cd_file):
                shutil.copy2(cd_file, dst_dir)
            else:
                # [TODO]: If it is not found, we need to create it.
                print(f"WARNING: clip_durations.tsv file not found for {ver} - {lc}")

    #
    # Main
    #

    # Loop all versions
    for ver in c.CV_VERSIONS:
        # Check if it exists in source (check "s1", if not there, it is nowhere)
        ver_dir = calc_dataset_prefix(ver)
        if not os.path.isdir(os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], ver_dir)):
            continue  # Does not exist, so skip

        # Create destination
        dst_dir = os.path.join(vc_dir_base, ver_dir)

        # Get a  list of available language codes
        lc_paths: list[str] = glob.glob(
            pathname=os.path.join(conf.SRC_BASE_DIR, c.ALGORITHMS[0], ver_dir, "*"),
            recursive=False,
        )
        lc_list: list[str] = [os.path.split(p)[-1] for p in lc_paths]
        lc_cnt: int = len(lc_list)

        g.processed_ver += 1
        g.total_lc += lc_cnt

        # print("=" * 80)
        # print(f"Processing {lc_cnt} locales in {cv_dir_name}")

        pbar_lc = tqdm(lc_list, desc=("   v" + ver)[-5:], total=lc_cnt, unit=" Locale")
        for lc in lc_list:
            handle_locale(lc)
            pbar_lc.update()

        pbar_lc.close()
        # pbar_ver.update()

    # done
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-compiler: Data Algorithms/Splits Collection Process ===")
    init_directories()
    main()

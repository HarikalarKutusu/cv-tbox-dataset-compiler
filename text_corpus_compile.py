#!/usr/bin/env python3

###########################################################################
# compile_text_corpus.py
#
# From cloned Common Voice Repo, Downloads all text corpus files from github for all locales.
#
# Use:
# python dl_text_corpora.py
#
#
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################

import re
import sys
import os
import shutil
import glob
import csv
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd


HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)


#
# Constants - TODO These should be arguments
#

CV_REPO: str = "C:\\GITREPO\\_AI_VOICE\\_CV\\common-voice\\server\\data"
# DSTBASE: str = "C:\\GITREPO\\_HK_GITHUB\\_cv_tbox\\cv-tbox-dataset-compiler\\datasets"

TC_COLS: "list[str]" = [
    "file", "sentence", "chars"  # , "words", "sentence_lower", "normalized", "valid"
]

# FINAL STRUCTURE AT DESTINATION
# clip-durations
#   <lc>
#       $clip_durations.tsv
# text-corpus
#   <lc>
#       $clip_durations.tsv
# voice-corpus
#   <cvver>                                             # eg: "cv-corpus-11.0-2022-09-21"
#       <lc>                                            # eg: "tr"
#           validated.tsv
#           invalidated.tsv
#           other.tsv
#           reported.tsv
#           <splitdir>
#               train.tsv
#               dev.tsv
#               test.tsv


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(fpath, header=True, index=False, encoding="utf-8",
              sep='\t', escapechar='\\', quoting=csv.QUOTE_NONE)
    print(f'Generated: {fpath} Records={df.shape[0]}')
    return True


# MAIN PROCESS
def main() -> None:
    print('=== Text-Corpora Compilation Process for cv-tbox-dataset-compiler ===')
    start_time: datetime = datetime.now()

    tc_base_dir: str = os.path.join(HERE, 'data', 'text-corpus')

    # Get a list of available language codes
    lc_paths: "list[str]" = glob.glob(
        os.path.join(CV_REPO, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        lc: str = os.path.split(lc_path)[1]
        lc_list.append(lc)

    # Create directory structure at the destination
    for lc in lc_list:
        os.makedirs(os.path.join(tc_base_dir, lc), exist_ok=True)

    # extra line is for progress line
    print(f'Processing text-corpus for {len(lc_list)} locales...\n')

    cnt: int = 0
    file_cnt: int = 0
    # Loop again to DL actual files
    for lc in lc_list:
        cnt += 1
        # print('\033[F' + ' ' * 120)
        # print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')
        print(f'Processing locale {cnt}/{len(lc_list)} : {lc}')

        src_path: str = os.path.join(CV_REPO, lc)
        dst_file: str = os.path.join(tc_base_dir, lc, "$text_corpus.tsv")

        # Create a DataFrame
        df: pd.DataFrame = pd.DataFrame(columns=TC_COLS)

        # get file list
        files: "list[str]" = glob.glob(
            os.path.join(src_path, '*'), recursive=False)

        # Process each file
        for text_file in files:
            file_cnt += 1
            with open(text_file, mode="r", encoding="utf-8") as fp:
                lines: "list[str]" = fp.readlines()
            pure_fn: str = os.path.split(text_file)[1]
            data: "list[dict[str,Any]]" = []
            # Process each line
            for line in lines:
                rec: "dict[str, Any]" = {
                    "file": pure_fn,
                    "sentence": line.strip("\n"),
                    "chars": len(line),
                }
                data.append(rec)
            data_df: pd.DataFrame = pd.DataFrame(data, columns=TC_COLS).reset_index(
                drop=True)
            df = pd.concat([df.loc[:], data_df]).reset_index(
                drop=True)  # type: ignore

        # write out to file
        df_write(df, dst_file)
        # print(df.head())
        # sys.exit()

    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'Finished compiling text-corpus for {len(lc_list)} locales from {file_cnt} files in {str(process_timedelta)} avg={process_seconds/len(lc_list)} sec/locale')


main()

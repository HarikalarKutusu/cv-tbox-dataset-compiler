#!/usr/bin/env python3
"""cv-tbox Dataset Compiler - Text-Corpus Compilation Phase"""
###########################################################################
# text_corpus_compile.py
#
# From cloned Common Voice Repo, get all text corpus files for all locales.
# Combine them and add some pre calculations.
#
# Use:
# python text_corpus_compile.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
import sys
import os
import glob
import multiprocessing as mp

# External dependencies
from tqdm import tqdm
import pandas as pd
import psutil

# Module
import const as c
import conf
from lib import (
    calc_dataset_prefix,
    df_write,
    get_locales_from_cv_dataset,
    git_checkout,
    git_clone_or_pull_all,
    init_directories,
    report_results,
)
from typedef import TextCorpusRec, Globals


# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
MAX_BATCH_SIZE: int = 5

g: Globals = Globals(
    total_ver=len(c.CV_VERSIONS),
    total_algo=len(c.ALGORITHMS),
)


def handle_locale(ver_lc: str) -> None:
    """Process to handle a single locale"""

    ver_dir: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]

    src_path: str = os.path.join(
        conf.CV_TBOX_CACHE, c.CLONES_DIRNAME, "common-voice", "server", "data", lc
    )
    tc_ver_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, ver_dir)
    dst_file: str = os.path.join(tc_ver_dir, lc, f"{c.TEXT_CORPUS_FN}.tsv")

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(columns=c.COLS_TEXT_CORPUS)

    # get file list
    files: list[str] = glob.glob(os.path.join(src_path, "*.txt"), recursive=False)

    # Process each file
    lines: list[str] = []
    for text_file in files:
        with open(text_file, mode="r", encoding="utf-8") as fp:
            lines: list[str] = fp.readlines()
        data: list[TextCorpusRec] = []
        fn: str = os.path.split(text_file)[1]
        # Process each line
        data = [TextCorpusRec(file=fn, sentence=line.strip("\n")) for line in lines]
        data_df: pd.DataFrame = pd.DataFrame(
            data, columns=c.COLS_TEXT_CORPUS
        ).reset_index(drop=True)
        df = pd.concat([df.loc[:], data_df]).reset_index(drop=True)  # type: ignore
    # end of files

    # write out to file
    df_write(df, dst_file)


def handle_version(inx: int, ver: str) -> None:
    """Handle a CV version"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    cutoff_date: str = c.CV_DATES[inx]
    print(f"=== HANDLE: v{ver} @ {cutoff_date} ===")
    git_checkout(c.CV_GITREC, cutoff_date)

    lc_list: list[str] = get_locales_from_cv_dataset(ver)
    total_locales: int = len(lc_list)

    # Filter out already processed
    ver_dir: str = calc_dataset_prefix(ver)
    tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, ver_dir)
    ver_lc_list: list[str] = [
        f"{ver_dir}|{lc}"
        for lc in lc_list
        if not os.path.isfile(os.path.join(tc_dir, lc, f"{c.TEXT_CORPUS_FN}.tsv"))
    ]
    num_locales: int = len(ver_lc_list)

    # [TODO] Handle remaining locales in multi-processing
    chunk_size: int = min(
        MAX_BATCH_SIZE,
        num_locales // PROC_COUNT + (0 if num_locales % PROC_COUNT == 0 else 1),
    )
    print(
        f"Total: {total_locales} Existing: {total_locales-num_locales} Remaining: {num_locales} "
        + f"Procs: {PROC_COUNT}  chunk_size: {chunk_size}..."
    )

    if num_locales > 0:
        with mp.Pool(PROC_COUNT) as pool:
            with tqdm(total=num_locales, desc="") as pbar:
                for _ in pool.imap_unordered(
                    handle_locale, ver_lc_list, chunksize=chunk_size
                ):
                    pbar.update()

    g.total_lc += total_locales
    g.processed_ver += 1
    g.processed_lc += num_locales
    g.skipped_exists += total_locales - num_locales


# MAIN PROCESS
def main() -> None:
    """Main function feeding the multi-processing pool"""

    # Make sure clones are current
    git_checkout(c.CV_GITREC)
    git_clone_or_pull_all()

    # Main loop for all versions
    for inx, ver in enumerate(c.CV_VERSIONS):
        handle_version(inx, ver)

    # done, revert to main and report
    git_checkout(c.CV_GITREC)
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-compiler: Text-Corpora Compilation Process ===")
    init_directories(HERE)
    main()

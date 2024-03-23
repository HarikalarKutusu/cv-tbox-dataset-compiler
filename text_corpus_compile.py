#!/usr/bin/env python3
"""cv-tbox Dataset Compiler - Text-Corpus Compilation Phase"""
###########################################################################
# text_corpus_compile.py
#
# From validated_sentences.tsv (after Common Voice v17.0), create/cache some pre-calculated measures
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
import multiprocessing as mp

# External dependencies
from tqdm import tqdm
import pandas as pd
import psutil
import cvutils as cvu

# Module
import const as c
import conf
from lib import (
    calc_dataset_prefix,
    df_read,
    df_write,
    get_locales_from_cv_dataset,
    git_checkout,
    git_clone_or_pull_all,
    init_directories,
    report_results,
)
from typedef import Globals


# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
MAX_BATCH_SIZE: int = 5

cv: cvu.CV = cvu.CV()
VALIDATORS: list[str] = cv.validators()
PHONEMISERS: list[str] = cv.phonemisers()
# ALPHABETS: list[str] = [str(p).split(os.sep)[-2] for p in cv.alphabets()]
# SEGMENTERS: list[str] = [str(p).split(os.sep)[-2] for p in cv.segmenters()]

g: Globals = Globals(
    total_ver=len(c.CV_VERSIONS),
    total_algo=len(c.ALGORITHMS),
)


def handle_locale(ver_lc: str) -> None:
    """Process to handle a single locale"""

    def handle_preprocess(df_base: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        """Get whole data and only process the unprocessed ones, returns the full result"""
        # if not forced, only work on new tsentences
        base_ids: list[str] = []
        if not conf.FORCE_CREATE_TC_STATS:
            base_ids = df_base["sentence_id"].to_list()
            df_new = df_new[~df_new["sentence_id"].isin(base_ids)]

        # pre-calc simpler values
        df_new["char_cnt"] = [
            len(s) if isinstance(s, str) else 0 for s in df_new["sentence"].to_list()
        ]

        # validator dependent
        if validator:
            df_new["normalized"] = [
                validator.validate(s) if isinstance(s, str) else None
                for s in df_new["sentence"].tolist()
            ]
            df_new["valid"] = [0 if n is None else 1 for n in df_new["normalized"].tolist()]
            df_new["tokens"] = [
                None if s is None else tokeniser.tokenise(s)
                for s in df_new["normalized"].tolist()
            ]
            df_new["word_cnt"] = [
                None if ww is None else len(ww) for ww in df_new["tokens"].tolist()
            ]

        # phonemiser dependent
        if phonemiser:
            df_new["phonemised"] = [
                phonemiser.phonemise(s) if isinstance(s, str) else None
                for s in df_new["sentence"].tolist()
                # for w in str(s).split(" ")
            ]
        # return with newly processed data added
        return pd.concat([df_base.astype(df_new.dtypes), df_new.astype(df_base.dtypes)])

    # handle_locale MAIN
    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    ver_dir: str = calc_dataset_prefix(ver)

    # precalc dir and file paths
    base_tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc)
    os.makedirs(base_tc_dir, exist_ok=True)
    base_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}.tsv")
    src_tc_val_file: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc, c.TC_VALIDATED_FILE
    )

    # cvu - do we have them?
    validator: cvu.Validator | None = (
        cvu.Validator(lc) if lc in VALIDATORS else None
    )
    phonemiser: cvu.Phonemiser | None = (
        cvu.Phonemiser(lc) if lc in PHONEMISERS else None
    )
    tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)

    # get existing base (already preprocessed) and new validated dataframes
    df_base: pd.DataFrame = pd.DataFrame(columns=c.COLS_TEXT_CORPUS)
    if os.path.isfile(base_tc_file):
        df_base = df_read(base_tc_file)
    df_tc_val: pd.DataFrame = df_read(src_tc_val_file)
    df_tc_val.reindex(columns=c.COLS_TEXT_CORPUS)  # add new columns

    df_write(handle_preprocess(df_base, df_tc_val), base_tc_file)  # write-out result


def handle_last_version() -> None:
    """Handle a CV version"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    ver: str = c.CV_VERSIONS[-1]
    cutoff_date: str = c.CV_DATES[-1]
    print(f"=== HANDLE: v{ver} @ {cutoff_date} ===")
    # git_checkout(c.CV_GITREC, cutoff_date)

    lc_list: list[str] = get_locales_from_cv_dataset(ver)
    total_locales: int = len(lc_list)

    # Filter out already processed
    tc_analysis_dir: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.TC_ANALYSIS_DIRNAME, calc_dataset_prefix(ver)
    )
    ver_lc_list: list[str] = [
        f"{ver}|{lc}"
        for lc in lc_list
        if not os.path.isdir(os.path.join(tc_analysis_dir, lc))
        or conf.FORCE_CREATE_TC_STATS
    ]
    num_locales: int = len(ver_lc_list)

    # Handle remaining locales in multi-processing
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

    # we don't need the git clones after v17.0

    # Make sure clones are current
    git_checkout(c.CV_GITREC)
    git_clone_or_pull_all()

    # Main loop for all versions - start from newest to get latest text corpus data
    # cv_version_reversed: list[str] = c.CV_VERSIONS.copy()
    # cv_version_reversed.reverse()
    # for inx, ver in enumerate(cv_version_reversed):
    #     handle_version(inx, ver)

    # Do it only for last version (after v17.0)
    handle_last_version()

    # done, revert to main and report
    # git_checkout(c.CV_GITREC)
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-compiler: Text-Corpora Compilation Process ===")
    init_directories(HERE)
    main()

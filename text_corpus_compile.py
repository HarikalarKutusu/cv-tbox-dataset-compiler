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
import glob
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
    get_cutoff_date,
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

#
# LAST VERSION HANDLERS
#


def handle_last_version_locale(ver_lc: str) -> None:
    """Process to handle a single locale in last version"""

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
            df_new["valid"] = [
                0 if n is None else 1 for n in df_new["normalized"].tolist()
            ]
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
    validator: cvu.Validator | None = cvu.Validator(lc) if lc in VALIDATORS else None
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
    """Handle last CV version"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    ver: str = c.CV_VERSIONS[-1]
    cutoff_date: str = c.CV_DATES[-1]
    print(f"=== HANDLE: v{ver} @ {cutoff_date} ===")
    # git_checkout(c.CV_GITREC, cutoff_date)

    lc_list: list[str] = get_locales_from_cv_dataset(ver)
    total_locales: int = len(lc_list)

    # Filter out already processed
    tc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)
    ver_lc_list: list[str] = [
        f"{ver}|{lc}"
        for lc in lc_list
        if not os.path.isfile(os.path.join(tc_base_dir, lc, f"{c.TEXT_CORPUS_FN}_{ver}.tsv"))
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
                    handle_last_version_locale, ver_lc_list, chunksize=chunk_size
                ):
                    pbar.update()

    g.total_lc += total_locales
    g.processed_ver += 1
    g.processed_lc += num_locales
    g.skipped_exists += total_locales - num_locales


#
# OLD VERSION HANDLERS
#


def handle_old_version_locale(ver_lc: str) -> None:
    """Process to handle a single locale in older versions"""

    # handle_locale MAIN
    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    ver_dir: str = calc_dataset_prefix(ver)

    # precalc dir and file paths
    base_tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc)
    base_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}.tsv")
    ver_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}.tsv")
    ver_vc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc)

    # get existing base (already preprocessed) and new validated dataframes
    df_base: pd.DataFrame = df_read(base_tc_file)
    df: pd.DataFrame = pd.DataFrame(columns=["sentence_id"])
    # For versions v17.0 and later, we just use the main text_corpora file - even current is generated
    if float(ver) >= 17.0:
        df = pd.DataFrame(df_base["sentence_id"]).drop_duplicates().dropna().sort_values("sentence_id")
        df_write(df, ver_tc_file)
        return

    # ELSE- For versions before v17.0, get the data from github clone + main buckets
    # These do not have "sentence_id" field, thus we need to use the "sentence" field to locate them

    # Get sentences from major buckets
    sentences: list[str] = []
    for bucket in c.MAIN_BUCKETS:
        ver_vc_bucket_file: str = os.path.join(ver_vc_dir, f"{bucket}.tsv")
        if os.path.isfile(ver_vc_bucket_file):
            df_temp: pd.DataFrame = df_read(ver_vc_bucket_file)
            sentences.extend(df_temp["sentence"].to_list())
    # remove duplicates
    sentences = list(set(sentences))

    # get sentences from git clone server/data/<lc>/*.txt
    git_data_path: str = os.path.join(
        conf.CV_TBOX_CACHE, "clones", "common-voice", "server", "data", lc
    )
    file_list: list[str] = glob.glob(
        os.path.join(git_data_path, "*.txt"), recursive=False
    )
    for fn in file_list:
        with open(fn, encoding="utf8") as fd:
            sentences.extend(fd.readlines())
    # re-remove duplicates
    sentences = list(set(sentences))

    # Now get the subset
    df_subset: pd.DataFrame = df_base[df_base["sentence"].isin(sentences)].drop_duplicates()
    df_sen_id: pd.DataFrame = df_subset["sentence_id"].to_frame().dropna().sort_values("sentence_id")

    df_write(df_sen_id, ver_tc_file)  # write-out result


def handle_older_version(ver: str) -> None:
    """Handle an older CV version - just keep sentence_id's in the result"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    cutoff_date: str = get_cutoff_date(ver)
    print(f"=== HANDLE: v{ver} @ {cutoff_date} ===")

    lc_list: list[str] = get_locales_from_cv_dataset(ver)
    total_locales: int = len(lc_list)

    # Filter out already processed
    base_tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)
    ver_lc_list: list[str] = [
        f"{ver}|{lc}"
        for lc in lc_list
        if not os.path.isfile(
            os.path.join(base_tc_dir, lc, f"{c.TEXT_CORPUS_FN}_{ver}.tsv")
        )
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
        git_checkout(c.CV_GITREC, cutoff_date)
        with mp.Pool(PROC_COUNT) as pool:
            with tqdm(total=num_locales, desc="") as pbar:
                for _ in pool.imap_unordered(
                    handle_old_version_locale, ver_lc_list, chunksize=chunk_size
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

    # Do it only for last version (after v17.0)
    handle_last_version()

    # Loop for versions, just to keep sentence_id info
    for ver in c.CV_VERSIONS:
        handle_older_version(ver)

    # done, revert to main and report
    git_checkout(c.CV_GITREC)
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-compiler: Text-Corpora Compilation Process ===")
    init_directories(HERE)
    main()

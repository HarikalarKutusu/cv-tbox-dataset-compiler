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
# from ast import literal_eval
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
    df_concat,
    df_read,
    df_write,
    get_cutoff_date,
    get_locales_from_cv_dataset,
    git_checkout,
    git_clone_or_pull_all,
    init_directories,
    report_results,
    sort_by_largest_file,
)
from typedef import Globals


# Globals
HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
MAX_BATCH_SIZE: int = 1
used_proc_count: int = conf.DEBUG_PROC_COUNT if conf.DEBUG else PROC_COUNT

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
# This is temporary for handling malformed validated_sentences.tsv of v17.0
#


def df_read_safe_tc_validated(fpath: str) -> pd.DataFrame:
    """Read in possibly malformed validated_sentences.tsv file"""
    lines_read: list[str] = []
    final_arr: list[list[str]] = []
    # read all data and split to lines
    with open(fpath, encoding="utf8") as fd:
        lines_read: list[str] = fd.read().splitlines()

    # first line is column names, so we skip it because we predefine them
    cur_source_line: int = 1
    while cur_source_line < len(lines_read):
        line: str = lines_read[cur_source_line].replace("\n", " ").replace("  ", " ")
        # if there are 6 columns, we are OK, else we append next line to this.
        if len(line.split("\t")) < len(c.FIELDS_TC_VALIDATED) and cur_source_line < len(
            lines_read
        ):
            if cur_source_line >= len(lines_read) - 1:
                break
            # not EOF
            next_line: str = lines_read[cur_source_line + 1].replace("\n", "")
            line += next_line
            cur_source_line += 1  # extra inc
        # normal inc
        cur_source_line += 1
        # add to final
        v: list[str] = line.split("\t")
        if len(v) == 6:
            final_arr.append(v)

    # _df: pd.DataFrame = pd.DataFrame(final_arr, columns=c.COLS_TC_VALIDATED).drop_duplicates()
    _df: pd.DataFrame = pd.DataFrame(
        final_arr, columns=c.FIELDS_TC_VALIDATED
    ).drop_duplicates()
    return _df


#
# LAST VERSION HANDLERS
#


def handle_last_version_locale(ver_lc: str) -> str:
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
            str(len(s)) if isinstance(s, str) else "0"
            for s in df_new["sentence"].to_list()
        ]

        # validator dependent
        if validator:
            df_new["normalized"] = [
                validator.validate(s) if isinstance(s, str) else None
                for s in df_new["sentence"].tolist()
            ]
            df_new["valid"] = [
                "0" if n is None else "1" for n in df_new["normalized"].tolist()
            ]
            df_new["tokens"] = [
                None if s is None else tokeniser.tokenise(s)
                for s in df_new["normalized"].tolist()
            ]
            df_new["word_cnt"] = [
                None if ww is None else str(len(ww)) for ww in df_new["tokens"].tolist()
            ]

        # phonemiser dependent
        if phonemiser:
            df_new["phonemised"] = [
                phonemiser.phonemise(s) if isinstance(s, str) else None
                for s in df_new["sentence"].tolist()
                # for w in str(s).split(" ")
            ]
        # return with newly processed data added
        return df_concat(df_base, df_new)

    # handle_locale MAIN
    # print("PROCESSING=", ver_lc)
    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    ver_dir: str = calc_dataset_prefix(ver)

    # cvu - do we have them?
    validator: cvu.Validator | None = cvu.Validator(lc) if lc in VALIDATORS else None
    phonemiser: cvu.Phonemiser | None = (
        cvu.Phonemiser(lc) if lc in PHONEMISERS else None
    )
    tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)

    # get existing base (already preprocessed) and new validated dataframes
    # df_base: pd.DataFrame = pd.DataFrame(columns=c.COLS_TEXT_CORPUS, dtype=c.DTYPES_TEXT_CORPUS)
    base_tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc)
    os.makedirs(base_tc_dir, exist_ok=True)
    base_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}.tsv")
    df_base: pd.DataFrame = pd.DataFrame(columns=c.FIELDS_TEXT_CORPUS)
    if os.path.isfile(base_tc_file):
        df_base = df_read(base_tc_file)

    # df_tc_val: pd.DataFrame = df_read(src_tc_val_file)
    src_tc_val_file: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc, c.TC_VALIDATED_FILE
    )
    df_tc_val: pd.DataFrame = df_read_safe_tc_validated(src_tc_val_file)
    df_tc_val = df_tc_val.reindex(columns=c.FIELDS_TEXT_CORPUS)  # add new columns

    # write-out result
    df_new_tc: pd.DataFrame = handle_preprocess(df_base, df_tc_val)
    if df_base.shape[0] != df_new_tc.shape[0]:
        df_write(df_new_tc, base_tc_file)
    # create index file for the last version
    df_write(
        df_new_tc["sentence_id"].to_frame().sort_values("sentence_id"),
        os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}.tsv"),
    )
    return ver_lc


def handle_last_version() -> None:
    """Handle last CV version"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    ver: str = c.CV_VERSIONS[-1]
    cutoff_date: str = c.CV_DATES[-1]
    print(f"=== HANDLE: v{ver} @ {cutoff_date} ===")
    # git_checkout(c.CV_GITREC, cutoff_date)

    lc_list: list[str] = get_locales_from_cv_dataset(ver)
    total_locales: int = len(lc_list)

    # Get list of new validated_sentences files, in reverse size order
    # then get a list of language codes in that order
    # This executes larger data first, so that multiprocessing is maximized
    pp: list[str] = glob.glob(
        os.path.join(
            HERE,
            c.DATA_DIRNAME,
            c.VC_DIRNAME,
            calc_dataset_prefix(ver),
            "**",
            c.TC_VALIDATED_FILE,
        )
    )
    lc_list = (
        [p.split(os.sep)[-2] for p in sort_by_largest_file(pp)]
        if not conf.DEBUG
        else conf.DEBUG_CV_LC
    )
    # Filter out already processed
    tc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)
    ver_lc_list: list[str] = [
        f"{ver}|{lc}"
        for lc in lc_list
        if not os.path.isfile(
            os.path.join(tc_base_dir, lc, f"{c.TEXT_CORPUS_FN}_{ver}.tsv")
        )
        or conf.FORCE_CREATE_TC_STATS
    ]
    num_locales: int = len(ver_lc_list)

    # Handle remaining locales in multi-processing
    chunk_size: int = min(
        MAX_BATCH_SIZE,
        num_locales // used_proc_count
        + (0 if num_locales % used_proc_count == 0 else 1),
    )
    print(
        f"Total: {total_locales} Existing: {total_locales-num_locales} Remaining: {num_locales} "
        + f"Procs: {used_proc_count}  chunk_size: {chunk_size}..."
    )

    if num_locales > 0:
        print(f"Processing: {[x.split(" | ")[1] for x in ver_lc_list]}")
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=num_locales, desc="Locales") as pbar:
                for res in pool.imap_unordered(
                    handle_last_version_locale, ver_lc_list, chunksize=chunk_size
                ):
                    pbar.write(f"Finished: {res}")
                    pbar.update()

    g.total_lc += total_locales
    g.processed_ver += 1
    g.processed_lc += num_locales
    g.skipped_exists += total_locales - num_locales


#
# OLD VERSION HANDLERS (Creates files which include only sentence_index)
#


def handle_old_version_locale(ver_lc: str) -> str:
    """Process to handle a single locale in older versions"""

    # handle_locale MAIN
    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    # ver_dir: str = calc_dataset_prefix(ver)

    # precalc dir and file paths
    base_tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc)
    base_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}.tsv")
    ver_tc_file: str = os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}.tsv")
    # ver_vc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc)

    # get existing base (already preprocessed) and new validated dataframes
    df_base: pd.DataFrame = pd.DataFrame(columns=c.FIELDS_TEXT_CORPUS)
    # NotImplementedError: Converting strings to list<item: string> is not implemented.
    # df_base = df_read(fpath=base_tc_file, dtype=c.FIELDS_TEXT_CORPUS)
    df_base = df_read(fpath=base_tc_file)
    # df_base = df_base[ df_base["is_used"] == "1" ][["sentence_id", "sentence"]] # we only need these
    df_base = df_base[df_base["is_used"] == 1]  # we only need these
    df_indexes: pd.DataFrame = pd.DataFrame(columns=["sentence_id"])
    # For versions v17.0 and later, we just use the main text_corpora file - even current is generated
    if float(ver) >= 17.0:
        df_indexes = (
            df_base["sentence_id"]
            .to_frame()
            .dropna()
            .drop_duplicates()
            .sort_values("sentence_id")
        )
        df_write(df_indexes, ver_tc_file)
        return ver_lc

    # ELSE- For versions before v17.0, get the data from github clone + main buckets
    # These do not have "sentence_id" field, thus we need to use the "sentence" field to locate them

    sentences: list[str] = []

    # get sentences from git clone server/data/<lc>/*.txt
    git_data_path: str = os.path.join(
        conf.CV_TBOX_CACHE, "clones", "common-voice", "server", "data", lc
    )
    file_list: list[str] = glob.glob(
        os.path.join(git_data_path, "*.txt"), recursive=False
    )
    for fn in file_list:
        with open(fn, encoding="utf8") as fd:
            sentences.extend(fd.read().splitlines())
    # make unique & get rid of new lines
    # sentences = [line.replace("\n", "").replace("\t", " ").replace("  ", " ") for line in sentences]
    sentences = [s for s in list(set(sentences)) if s]

    # [FIXME] The following does not work as the sentences are post-manipulated by CorporaCreator
    # Get sentences from major buckets
    # for bucket in c.MAIN_BUCKETS:
    #     ver_vc_bucket_file: str = os.path.join(ver_vc_dir, f"{bucket}.tsv")
    #     if os.path.isfile(ver_vc_bucket_file):
    #         df_temp: pd.DataFrame = df_read(ver_vc_bucket_file)
    #         sentences.extend(df_temp["sentence"].to_list())
    # # remove duplicates
    # sentences = list(set(sentences))

    # now get a subset
    df_subset: pd.DataFrame = df_base[df_base["sentence"].isin(sentences)]
    # print(df_subset.shape[0])
    df_indexes = (
        df_subset["sentence_id"]
        .to_frame()
        .dropna()
        .drop_duplicates()
        .sort_values("sentence_id")
    )
    # debug df's
    # df_sentences: pd.DataFrame = pd.DataFrame(sentences, columns=["sentence"])
    # df_exluded: pd.DataFrame = df_base[~df_base["sentence"].isin(sentences)]
    # df_write(df_sentences, os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}_sentences.tsv"))
    # df_write(df_subset, os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}_subset.tsv"))
    # df_write(df_exluded, os.path.join(base_tc_dir, f"{c.TEXT_CORPUS_FN}_{ver}_excluded.tsv"))

    # write-out result
    df_write(df_indexes, ver_tc_file)
    return ver_lc


def handle_older_version(ver: str) -> None:
    """Handle an older CV version - just keep sentence_id's in the result"""

    # Get the repo at cutoff date ([TODO] Need to compile real cut-off dates)
    cutoff_date: str = get_cutoff_date(ver)
    print(f"=== HANDLE INDEXING: v{ver} @ {cutoff_date} ===")

    lc_list: list[str] = (
        get_locales_from_cv_dataset(ver) if not conf.DEBUG else conf.DEBUG_CV_LC
    )
    total_locales: int = len(lc_list)

    # Get list of existing processed text corpus files, in reverse size order
    # then get a list of language codes in that order
    # This assumes that the larger the latest TC, the larger data we will have in previous versions,
    # so that multiprocessing is maximized
    pp: list[str] = glob.glob(
        os.path.join(
            HERE, c.DATA_DIRNAME, c.TC_DIRNAME, "**", f"{c.TEXT_CORPUS_FN}.tsv"
        )
    )
    lc_complete_list: list[str] = [
        p.split(os.sep)[-2] for p in sort_by_largest_file(pp)
    ]
    lc_list = (
        [lc for lc in lc_complete_list if lc in lc_list]
        if not conf.DEBUG
        else conf.DEBUG_CV_LC
    )

    # Get lc list and filter out already processed
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
        num_locales // used_proc_count
        + (0 if num_locales % used_proc_count == 0 else 1),
    )
    print(
        f"Total: {total_locales} Existing: {total_locales-num_locales} Remaining: {num_locales} "
        + f"Procs: {used_proc_count}  chunk_size: {chunk_size}..."
    )

    if num_locales > 0:
        print(f"Processing: {[x.split(" | ")[1] for x in ver_lc_list]}")
        git_checkout(c.CV_GITREC, cutoff_date)
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=num_locales, desc="Locales") as pbar:
                for res in pool.imap_unordered(
                    handle_old_version_locale, ver_lc_list, chunksize=chunk_size
                ):
                    pbar.write(f"Finished: {res}")
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

    # Do it only for last version (after v17.0)
    handle_last_version()

    # Loop for versions in reverse, just to keep sentence_id info
    rev_cv_versions: list[str] = (
        c.CV_VERSIONS if not conf.DEBUG else conf.DEBUG_CV_VER
    ).copy()
    rev_cv_versions.reverse()
    for ver in rev_cv_versions:
        handle_older_version(ver)

    # done, revert to main and report
    git_checkout(c.CV_GITREC)
    report_results(g)


if __name__ == "__main__":
    print("=== cv-tbox-dataset-compiler: Text-Corpora Compilation Process ===")
    init_directories(HERE)
    main()

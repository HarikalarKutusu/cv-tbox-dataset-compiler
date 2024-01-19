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
from typing import cast
from collections import Counter
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

cv: cvu.CV = cvu.CV()
# [TODO] Remove these after the portability PR gets included in the release
# ALPHABETS: list[str] = [str(p).split(os.sep)[-2] for p in cv.alphabets()]
PHONEMISERS: list[str] = [str(p).split(os.sep)[-2] for p in cv.phonemisers()]
# SEGMENTERS: list[str] = [str(p).split(os.sep)[-2] for p in cv.segmenters()]
VALIDATORS: list[str] = [str(p).split(os.sep)[-2] for p in cv.validators()]

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
    dst_tokens_file: str = os.path.join(tc_ver_dir, lc, f"{c.TOKENS_FN}.tsv")
    dst_graphemes_file: str = os.path.join(tc_ver_dir, lc, f"{c.GRAPHEMES_FN}.tsv")
    dst_phonemes_file: str = os.path.join(tc_ver_dir, lc, f"{c.PHONEMES_FN}.tsv")

    # cvu
    # do we have them?
    validator: cvu.Validator = cvu.Validator(lc) if lc in VALIDATORS else None
    phonemiser: cvu.Phonemiser = cvu.Phonemiser(lc) if lc in PHONEMISERS else None
    tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)

    token_counter: Counter = Counter()
    grapheme_counter: Counter = Counter()
    phoneme_counter: Counter = Counter()

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(columns=c.COLS_TEXT_CORPUS)

    # get file list
    files: list[str] = glob.glob(os.path.join(src_path, "*.txt"), recursive=False)

    # Process each file
    lines: list[str] = []
    line: str = ""
    for text_file in files:
        with open(text_file, mode="r", encoding="utf-8") as fp:
            lines: list[str] = fp.readlines()
        data: list[TextCorpusRec] = []
        fn: str = os.path.split(text_file)[1]
        # Process each line
        for line in lines:
            line = line.strip("\n")
            rec: TextCorpusRec = TextCorpusRec(file=fn, sentence=line, chars=len(line))

            grapheme_counter.update(line)

            if phonemiser:
                phons: list[str] = [
                    phonemiser.phonemise(w) for w in rec.sentence.split(" ")
                ]
                if phons:
                    phoneme_counter.update("".join(p for p in phons if p))

            tokens: list[str] = []
            if validator:
                rec.normalized = validator.validate(rec.sentence)
                if rec.normalized:
                    tokens = cast(list[str], tokeniser.tokenise(rec.normalized))
                    rec.words = len(tokens)
                    token_counter.update(tokens)
                else:
                    rec.valid = 0

            data.append(rec)
        # end of file
        data_df: pd.DataFrame = pd.DataFrame(
            data, columns=c.COLS_TEXT_CORPUS
        ).reset_index(drop=True)
        df = pd.concat([df.loc[:], data_df]).reset_index(drop=True)  # type: ignore
    # end of files

    # write out to file
    df_write(df, dst_file)

    # graphemes df
    df: pd.DataFrame = pd.DataFrame(
        grapheme_counter.items(), columns=c.COLS_GRAPHEMES
    ).reset_index(drop=True)
    df.sort_values("count", ascending=False, inplace=True)
    df_write(df, dst_graphemes_file)

    # tokens df
    if validator:
        df: pd.DataFrame = pd.DataFrame(
            token_counter.items(), columns=c.COLS_TOKENS
        ).reset_index(drop=True)
        df.sort_values("count", ascending=False, inplace=True)
        df_write(df, dst_tokens_file)

    # phonemes df
    if phonemiser:
        df: pd.DataFrame = pd.DataFrame(
            phoneme_counter.items(), columns=c.COLS_PHONEMES
        ).reset_index(drop=True)
        df.sort_values("count", ascending=False, inplace=True)
        df_write(df, dst_phonemes_file)


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

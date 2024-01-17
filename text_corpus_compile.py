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
from datetime import datetime
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
from lib import dec3, df_write
from typedef import TextCorpusRec

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


def get_cv_repo() -> None:
    """Clones or pulls/updates the Common Voice repo"""
    # [TODO]
    return


def get_cv_repo_at_cutoff() -> None:
    """Gets the state of CV repo at the cut-off date"""
    # [TODO]
    return


def handle_locale(lc: str) -> None:
    """Process to handle a single locale"""

    # print('\033[F' + ' ' * 120)
    # print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')
    # print(f"Processing locale: {lc}")

    tc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)

    src_path: str = os.path.join(conf.CV_REPO, lc)
    dst_file: str = os.path.join(tc_base_dir, lc, f"{c.TEXT_CORPUS_FN}.tsv")
    dst_tokens_file: str = os.path.join(tc_base_dir, lc, f"{c.TOKENS_FN}.tsv")
    dst_graphemes_file: str = os.path.join(tc_base_dir, lc, f"{c.GRAPHEMES_FN}.tsv")
    dst_phonemes_file: str = os.path.join(tc_base_dir, lc, f"{c.PHONEMES_FN}.tsv")

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
    files: list[str] = glob.glob(os.path.join(src_path, "*"), recursive=False)

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


def handle_version() -> None:
    """Handle a CV version"""
    # [TODO] Get the repo at cutoff date
    # [TODO] Get all locales for that version
    # [TODO] Filter out existing
    # [TODO] Handle remaining locales in multi-processing


# MAIN PROCESS
def main() -> None:
    """Main function feeding the multi-processing pool"""

    print("=== Text-Corpora Compilation Process for cv-tbox-dataset-compiler ===")
    start_time: datetime = datetime.now()

    tc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)

    # Get a list of available language codes
    lc_paths: list[str] = glob.glob(os.path.join(conf.CV_REPO, "*"), recursive=False)
    lc_list: list[str] = sorted(
        [os.path.split(p)[-1] for p in lc_paths if os.path.isdir(p)]
    )

    # for lc_path in lc_paths:
    #     if os.path.isdir(lc_path):  # ignore files
    #         lc_list.append(os.path.split(lc_path)[1])

    # Create directory structure at the destination
    for lc in lc_list:
        os.makedirs(os.path.join(tc_base_dir, lc), exist_ok=True)

    # extra line is for progress line
    num_locales: int = len(lc_list)
    chunk_size: int = min(
        MAX_BATCH_SIZE,
        num_locales // PROC_COUNT + 0 if num_locales % PROC_COUNT == 0 else 1,
    )
    print(
        f"Processing text-corpora for {num_locales} locales in {PROC_COUNT} processes with chunk_size {chunk_size}...\n"
    )

    with mp.Pool(PROC_COUNT) as pool:
        with tqdm(total=num_locales, desc="") as pbar:
            for result in pool.imap_unordered(
                handle_locale, lc_list, chunksize=chunk_size
            ):
                pbar.update()

    # done
    process_seconds: float = (datetime.now() - start_time).total_seconds()
    print(
        f"Finished compiling text-corpus for {num_locales} locales in {dec3(process_seconds)}"
        + f" avg={dec3(process_seconds/num_locales)} sec/locale"
    )


if __name__ == "__main__":
    main()

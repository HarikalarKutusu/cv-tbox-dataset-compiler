#!/usr/bin/env python3

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

import sys, os, glob, csv
from datetime import datetime, timedelta
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

# MultiProcessing
import multiprocessing as mp
import psutil

# Common Voice Utilities
import cvutils as cvu

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

import const
import config as conf


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(fpath, header=True, index=False, encoding="utf-8",
              sep='\t', escapechar='\\', quoting=csv.QUOTE_NONE)
    print(f'Generated: {fpath} Records={df.shape[0]}')
    return True


def handle_locale(lc: str) -> None:
    # print('\033[F' + ' ' * 120)
    # print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')
    print(f'Processing locale: {lc}')

    tc_base_dir: str = os.path.join(HERE, 'data', 'text-corpus')

    src_path: str = os.path.join(conf.CV_REPO, lc)
    dst_file: str = os.path.join(tc_base_dir, lc, "$text_corpus.tsv")
    dst_tokens_file: str = os.path.join(tc_base_dir, lc, "$tokens.tsv")

    cv: cvu.CV = cvu.CV()
    supported: bool = False
    validator: cvu.Validator = cvu.Validator(lc)
    tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)
    for val in cv.validators():
        if lc == os.path.split(os.path.split(val)[0])[1]:
            supported = True

    tokenCounter: Counter = Counter()

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(columns=const.COLS_TEXT_CORPUS)

    # get file list
    files: "list[str]" = glob.glob(
        os.path.join(src_path, '*'), recursive=False)

    # Process each file
    for text_file in files:
        with open(text_file, mode="r", encoding="utf-8") as fp:
            lines: "list[str]" = fp.readlines()
        pure_fn: str = os.path.split(text_file)[1]
        data: "list[dict[str,Any]]" = []
        # Process each line
        for line in lines:
            line: str = line.strip("\n")
            valid: int = 1
            norm = None
            tokens = []
            valid = 1
            if supported:
                norm = validator.validate(line)
                if norm:
                    tokens: "list[str]" = tokeniser.tokenise(norm)
                    tokenCounter.update(tokens)
                else:
                    valid = 0

            # "file", "sentence", "lower", "normalized", "chars", "words" #, 'valid'
            rec: "dict[str, Any]" = {
                "file": pure_fn,
                "sentence": line,
                "normalized": norm,
                "chars": len(line),
                "words": len(tokens),
                "valid": valid
            }
            data.append(rec)
            # print(rec)
        # end of file
        data_df: pd.DataFrame = pd.DataFrame(data, columns=const.COLS_TEXT_CORPUS).reset_index(
            drop=True)
        df = pd.concat([df.loc[:], data_df]).reset_index(
            drop=True)  # type: ignore
    # end of files

    # write out to file
    df_write(df, dst_file)
    # tokens df
    df: pd.DataFrame = pd.DataFrame(tokenCounter.items(
    ), columns=const.COLS_TOKENS).reset_index(drop=True)
    df.sort_values("count", ascending=False, inplace=True)
    df_write(df, dst_tokens_file)

    # print(df.head())
    # sys.exit()


# MAIN PROCESS
def main() -> None:
    print('=== Text-Corpora Compilation Process for cv-tbox-dataset-compiler ===')
    start_time: datetime = datetime.now()

    tc_base_dir: str = os.path.join(HERE, 'data', 'text-corpus')

    # Get a list of available language codes
    lc_paths: "list[str]" = glob.glob(
        os.path.join(conf.CV_REPO, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        if os.path.isdir(lc_path):  # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    # Create directory structure at the destination
    for lc in lc_list:
        os.makedirs(os.path.join(tc_base_dir, lc), exist_ok=True)

    # extra line is for progress line
    print(f'Processing text-corpora for {len(lc_list)} locales...\n')

    with mp.Pool(psutil.cpu_count(logical=False) - 1) as pool:
        pool.map(handle_locale, lc_list)

    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'Finished compiling text-corpus for {len(lc_list)} locales in {str(process_timedelta)} avg={process_seconds/len(lc_list)} sec/locale')


if __name__ == '__main__':
    main()

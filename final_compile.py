#!/usr/bin/env python3

###########################################################################
# compile_stats.py
#
# From all data, compile result statistics data to be used in
# cv-tbox-dataset-analyzer
#
# Use:
# python compile_stats.py
#
#
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################

from genericpath import isfile
import sys
import os
import glob
import csv
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import const

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants
#

# Program parameters
VERBOSE: bool = False
FAIL_ON_NOT_FOUND: bool = True


# Support Matrix DataFrame
rev_versions: "list[str]" = const.CV_VERSIONS.copy()
rev_versions.reverse()
COLS_SUPPORT_MATRIX: "list[str]" = ['lc', 'lang']
COLS_SUPPORT_MATRIX.extend(rev_versions)
df_support_matrix: pd.DataFrame = pd.DataFrame(
    index=const.ALL_LOCALES,
    columns=COLS_SUPPORT_MATRIX,
    )


# STRUCTURE AT SOURCE
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

# FINAL STRUCTURE AT DESTINATION
# results
#   tsv
#     <lc>
#       <cvver>.tsv                 # eg: "tr_v11.0.tsv"
#   json
#     <lc>
#       <cvver>.json                # eg: "tr_v11.0.json" => copy these json files into cv-tbox-dataset-analyzer (600+ files)


###################################################
# Tooling
########################################################

def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f'FATAL: File {fpath} cannot be located!')
        if FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines='skip',
        quotechar='"',
        quoting=csv.QUOTE_NONE,
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(
        fpath,
        header=True,
        index=False,
        encoding="utf-8",
        sep='\t',
        escapechar='\\',
        quoting=csv.QUOTE_NONE,
        # float_format="%.4f"
    )
    if VERBOSE:
        print(f'Generated: {fpath} Records={df.shape[0]}')
    return True


def df_int_convert(x: pd.Series) -> Any:
    try:
        return x.astype(int)
    except:
        return x


def list2str(lst: "list[Any]") -> str:
    return const.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: "list[list[Any]]") -> str:
    return const.SEP_ROW.join(list2str(x) for x in arr)

########################################################
# Split Stats
########################################################


def split_stats(cv_idx: int) -> None:
    global df_support_matrix
    #
    # Handle one split, this is where calculations happen
    #
    # The default column structure of CV dataset splits is as follows
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment

    # we have as input:
    # 'version', 'locale', 'algorithm', 'split'

    # now, do calculate some statistics. We need:
    # 'clips', 'unique_voices', 'unique_sentences', 'duplicate_sentences',
    # 'genders_nodata', 'genders_male', 'genders_female', 'genders_other',
    # 'ages_nodata', 'ages_teens', 'ages_twenties', 'ages_thirties', 'ages_fourties', 'ages_fifties', 'ages_sixties', 'ages_seventies', 'ages_eighties', 'ages_nineties'

    def handle_split(ver: str, lc: str, algorithm: str, split: str, fpath: str) -> "dict[str, Any]":
        """Processes a single split and return calculated values"""

        nonlocal df_clip_durations, df_text_corpus

        # def _get_row_total(pt: pd.DataFrame, lbl: str) -> int:
        #     # return int(pd.to_numeric(pt.loc[lbl, 'TOTAL'])) if lbl in list(pt.index.values) else int(0)
        #     return int(pd.to_numeric(pt.loc[lbl, 'TOTAL']))

        # def _get_col_total(pt: pd.DataFrame, lbl: str) -> int:
        #     # return int(pd.to_numeric(pt.loc['TOTAL', lbl])) if lbl in list(pt.columns.values) else int(0)
        #     return pt.at['TOTAL', lbl]

        # === START ===
        # print("ver=", ver, "lc=", lc, "algorithm=", algorithm, "split=", split)

        # Read in DataFrames
        df: pd.DataFrame = df_read(fpath)

        # Do nothing, if there is no data
        if df.shape[0] == 0:
            results = {
                'ver':  ver,
                'lc':   lc,
                'alg':  algorithm,
                'sp':   split,
            }
            return results

        # Replace NA with NODATA
        df.fillna(value=const.NODATA, inplace=True)
        # add lowercase sentence column
        df['sentence_lower'] = df['sentence'].str.lower()

        #
        # DURATIONS
        #

        # Connect with duration table
        df["duration"] = df["path"].map(df_clip_durations["duration"])
        # Calc duration agregate values
        ser: pd.Series = df["duration"]
        if df_clip_durations.shape[0] > 0:
            duration_total: float = ser.sum()
            duration_mean: float = ser.mean()
            duration_median: float = ser.median()
            # Calc duration distribution
            arr: np.ndarray = np.fromiter(df["duration"].dropna().apply(
                int).reset_index(drop=True).to_list(), int)
            hist: list[list[int]] = np.histogram(arr, bins=const.BINS_DURATION)
            duration_freq: "list[int]" = hist[0]
            duration_bins: "list[int]" = hist[1]
        else: # No Duration data, set illegal defaults and continue
            duration_total: float = -1
            duration_mean: float = -1
            duration_median: float = -1
            duration_freq: "list[int]" = []
            duration_bins: "list[int]" = []

        #
        # VOICES
        #
        voice_counts: pd.DataFrame = df["client_id"].value_counts(
        ).to_frame().reset_index()
        voice_counts.rename(
            columns={"index": "voice", "client_id": "recordings"}, inplace=True)
        voice_mean: float = voice_counts["recordings"].mean()
        voice_median: float = voice_counts["recordings"].median()
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(voice_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_VOICES)
        voice_freq: "list[int]" = hist[0]
        voice_bins: "list[int]" = hist[1]

        #
        # SENTENCES
        #
        sentence_counts: pd.DataFrame = df["sentence"].value_counts(
        ).to_frame().reset_index()
        # print(sentence_counts)
        # sys.exit()

        sentence_counts.rename(
            columns={"index": "sentence", "sentence": "recordings"}, inplace=True)
        sentence_mean: float = sentence_counts["recordings"].mean()
        sentence_median: float = sentence_counts["recordings"].median()
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(sentence_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_SENTENCES)
        sentence_freq: "list[int]" = hist[0]
        sentence_bins: "list[int]" = hist[1]

        # print(duration_freq)
        # print(voice_freq)
        # print(sentence_freq)
        # sys.exit()

        # basic measures
        clips_cnt: int = df.shape[0]
        unique_voices: int = df['client_id'].unique().shape[0]
        unique_sentences: int = df['sentence'].unique().shape[0]
        unique_sentences_lower: int = df['sentence_lower'].unique().shape[0]
        duplicate_sentence_cnt: int = clips_cnt - unique_sentences
        duplicate_sentence_cnt_lower: int = clips_cnt - unique_sentences_lower

        # get a pt for all demographics
        _pt: pd.DataFrame = pd.pivot_table(
            df, values='path', index=['age'], columns=['gender'], aggfunc='count',
            fill_value=0, dropna=False, margins=True, margins_name='TOTAL'
        )
        # print(_pt)
        pt_ages: list[str] = const.CV_AGES.copy()
        pt_ages.append("TOTAL")
        pt_genders: list[str] = const.CV_GENDERS.copy()
        pt_genders.append("TOTAL")
        _pt = _pt.reindex(pt_ages, axis=0)
        _pt = _pt.reindex(pt_genders, axis=1)
        _pt = _pt.fillna(value=0).astype(int)
        # print(_pt)

        _males: int = _pt.at['TOTAL', 'male'].item()  # _get_col_total(_pt, 'male')
        _females: int = _pt.at['TOTAL', 'female'].item()

        results: dict[str, Any] = {
            'ver':          ver,
            'lc':           lc,
            'alg':          algorithm,
            'sp':           split,

            'clips':        clips_cnt,
            'uq_v':         unique_voices,
            'uq_s':         unique_sentences,
            'uq_sl':        unique_sentences_lower,
            'dup_s':        duplicate_sentence_cnt,
            'dup_sl':       duplicate_sentence_cnt_lower,
            # 'dup_r':        duplicate_sentence_cnt / unique_sentences,
            # 'dup_rl':       duplicate_sentence_cnt_lower / unique_sentences_lower,

            # 'g_m':          _males,
            # 'g_f':          _females,
            # 'g_o':          _pt.at['TOTAL', 'other'].item(),
            # 'g_nd':         _pt.at['TOTAL', const.NODATA].item(),

            # 'a_t':          _pt.at['teens', 'TOTAL'].item(),
            # 'a_20':         _pt.at['twenties', 'TOTAL'].item(),
            # 'a_30':         _pt.at['thirties', 'TOTAL'].item(),
            # 'a_40':         _pt.at['fourties', 'TOTAL'].item(),
            # 'a_50':         _pt.at['fifties', 'TOTAL'].item(),
            # 'a_60':         _pt.at['sixties', 'TOTAL'].item(),
            # 'a_70':         _pt.at['seventies', 'TOTAL'].item(),
            # 'a_80':         _pt.at['eighties', 'TOTAL'].item(),
            # 'a_90':         _pt.at['nineties', 'TOTAL'].item(),
            # 'a_nd':         _pt.at[const.NODATA, 'TOTAL'].item(),

            # 'g_f_m_ratio':        _females / _males if _males > 0 else -1,

            # TODO Add More columns

            # Duration
            'dur_total':    duration_total,
            'dur_mean':     duration_mean,
            'dur_median':   duration_median,
            'dur_freq':     list2str(duration_freq),

            # Recordings per Voice
            'v_mean':       voice_mean,
            'v_median':     voice_median,
            'v_freq':       list2str(voice_freq),

            # Recordings per Sentence
            's_mean':       sentence_mean,
            's_median':     sentence_median,
            's_freq':       list2str(sentence_freq),

            # Demographics distribution
            'dem_table':    arr2str(_pt.to_numpy(int).tolist()),

            # Tables & lists
            # 'dur_bins':     list2str(duration_bins),
            # 'v_bins':       list2str(voice_bins),
            # 's_bins':       list2str(sentence_bins),

        }

        # print(results)
        # sys.exit()

        return results
    # END handle_split

    # --------------------------------------------
    # START main process for a single CV version
    # --------------------------------------------

    # Calc CV_DIR
    ver: str = const.CV_VERSIONS[cv_idx]
    cv_dir_name: str = "cv-corpus-" + ver + "-" + \
        const.CV_DATES[cv_idx]  # TODO Handle older version naming
    # Calc voice-corpus directory
    cv_dir: str = os.path.join(HERE, "data", "voice-corpus", cv_dir_name)

    # Get a list of available language codes
    lc_paths: "list[str]" = glob.glob(
        os.path.join(cv_dir, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        lc_list.append(os.path.split(lc_path)[1])

    # Loop all locales
    cnt: int = 0
    res: "list[dict[str,Any]]" = []
    for lc in lc_list:
        cnt += 1
        print('\033[F' + ' ' * 80)
        print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')

        # Source directories
        vc_dir: str = os.path.join(cv_dir, lc)
        tc_dir: str = os.path.join(HERE, "data", "text-corpus", lc)
        cd_dir: str = os.path.join(HERE, "data", "clip-durations", lc)

        # Create destinations if thet do not exist
        tsv_path: str = os.path.join(HERE, "data", 'results', 'tsv', lc)
        json_path: str = os.path.join(HERE, "data", 'results', 'json', lc)
        os.makedirs(tsv_path, exist_ok=True)
        os.makedirs(json_path, exist_ok=True)

        #
        # First Handle Splits in voice-corpus
        #

        # Load general DF's if they exist, else initialize

        # Clip Durations
        cd_file: str = os.path.join(cd_dir, '$clip_durations.tsv')
        if os.path.isfile(cd_file):
            df_clip_durations: pd.DataFrame = df_read(cd_file).set_index("clip")
        else:
            print(f'WARNING: No duration data for {lc}\n')
            df_clip_durations: pd.DataFrame = pd.DataFrame(
                columns=const.COLS_CLIP_DURATIONS).set_index("clip")

        # Text Corpus
        tc_file: str = os.path.join(tc_dir, '$text_corpus.tsv')
        if os.path.isfile(tc_file):
            df_text_corpus: pd.DataFrame = df_read(tc_file) #.set_index("sentence")
        else:
            print(f'WARNING: No text-corpus for {lc}\n')
            df_text_corpus: pd.DataFrame = pd.DataFrame(columns=const.COLS_TEXT_CORPUS) #.set_index("sentence")

        # Add to support Matrix
        df_support_matrix.at[lc, ver] = "s1"

        # MAIN SPLITS

        # def handle_split(ver: str, lc: str, algorithm: str, split: str, fpath: str) -> "dict[str, Any]":
        for sp in const.MAIN_SPLITS:
            res.append(handle_split(
                ver=ver,
                lc=lc,
                algorithm="",
                split=sp,
                fpath=os.path.join(vc_dir, sp + '.tsv')
            ))
        
        # SPLITTING ALGO SPECIFIC (inc default splits)

        for algo in const.ALGORITHMS:
            if os.path.isdir(os.path.join(vc_dir, algo)):
                df_support_matrix.at[lc,
                                     ver] = df_support_matrix.at[lc, ver] + "-" + algo
                for sp in const.TRAINING_SPLITS:
                    res.append(handle_split(
                        ver=ver,
                        lc=lc,
                        algorithm=algo,
                        split=sp,
                        fpath=os.path.join(vc_dir, algo, sp + '.tsv')
                    ))

        # Create DataFrames
        df: pd.DataFrame = pd.DataFrame(res)
        df_write(df, os.path.join(tsv_path, f"{lc}_{ver}_splits.tsv"))
        df.to_json(os.path.join(json_path, f"{lc}_{ver}_splits.json"),
                   orient='table', index=False)

        # sys.exit()


########################################################
# Reported Stats
########################################################


# def reported_stats(lc_list: "list[str]") -> None:
#     pass

########################################################
# Text-Corpus Stats
########################################################


# def tc_stats(lc_list: "list[str]") -> None:
#     pass


########################################################
# MAIN PROCESS
########################################################
def main() -> None:
    global df_support_matrix

    print('=== Statistics Compilation Process for cv-tbox-dataset-analyzer ===')
    start_time: datetime = datetime.now()

    cnt_datasets: int = 0
    # Loop all versions
    for cv_idx, cv_ver in enumerate(const.CV_VERSIONS):
        # Calc CV_DIR
        cv_dir_name: str = "cv-corpus-" + cv_ver + "-" + const.CV_DATES[cv_idx]
        # Check if it exists in voice-corpus source
        vc_dir: str = os.path.join(HERE, 'data', 'voice-corpus', cv_dir_name)
        # print('CHECK:', vc_dir)
        if not os.path.isdir(vc_dir):  # if it does not exist, just skip that
            continue

        # Start Process
        print(f'\nProcessing {cv_dir_name}\n')   # extra line is for progress line
        split_stats(cv_idx)

        # Reported Stats
        # extra line is for progress line
        # print("START: Reported Statistics...\n")
        # reported_stats(lc_list)

        # Text-corpus Stats
        # extra line is for progress line
        # print("START: Text-Corpus Statistics...\n")
        # reported_stats(lc_list)

    # Save Support Matrix
    df_write(df_support_matrix, os.path.join(HERE, 'data', 'results', 'tsv', '$support_matrix.tsv'))
    df_support_matrix.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$support_matrix.json'),
        orient='table', index=False)
    # done
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    # print(f'Finished compiling statistics for {cnt_datasets} datasets in {str(process_timedelta)}, avg={process_seconds/cnt_datasets} sec/dataset.')


main()

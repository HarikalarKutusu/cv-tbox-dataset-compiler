#!/usr/bin/env python3

###########################################################################
# final_compile.py
#
# From all data, compile result statistics data to be used in
# cv-tbox-dataset-analyzer
#
# Use:
# python final_compile.py
#
#
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

import sys
import os
import glob
import csv
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# MULTIPROCESSING
import multiprocessing as mp
import psutil

# This package
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
PROC_COUNT: int = psutil.cpu_count(logical=False) - 1

cnt_datasets: int = 0

# Debug & Limiters
DEBUG: bool = False
DEBUG_PROC_COUNT: int = 1
DEBUG_CV_VER: "list[str]" = ['11.0']
DEBUG_CV_LC: "list[str]" = ['tr']


########################################################
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


def split_stats(cv_idx: int) -> "list[dict[str,Any]]":
    global cnt_datasets

    #
    # Handle one split, this is where calculations happen
    #
    # The default column structure of CV dataset splits is as follows
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment

    # we have as input:
    # 'version', 'locale', 'algorithm', 'split'

    # now, do calculate some statistics...
    def handle_split(ver: str, lc: str, algorithm: str, split: str, fpath: str) -> "dict[str, Any]":
        """Processes a single split and return calculated values"""
        nonlocal df_clip_durations

        #
        # find_fixes
        #
        def find_fixes(df_split: pd.DataFrame) -> "list[str]":
            """Finds fixable demographic info from the split and returns a string"""

            # df is local dataframe which will keep records only necessay columns with some additional columns
            df: pd.DataFrame = df_split.copy().reset_index(drop=True)
            df['v_enum'], v_unique = pd.factorize(df['client_id'])    # add an enumaration column for client_id's, more memory efficient
            df['p_enum'], p_unique = pd.factorize(df['path'])         # add an enumaration column for recordings, more memory efficient
            df = df[["v_enum", "age", "gender", "p_enum"]].fillna(const.NODATA).reset_index(drop=True)

            # prepare empty results
            fixes: pd.DataFrame = pd.DataFrame(columns=df.columns).reset_index(drop=True)
            dem_fixes_recs: str = ""
            dem_fixes_voices: str = ""

            # get unique voices with multiple demographic values
            df_counts: pd.DataFrame = df[["v_enum", "age", "gender"]].drop_duplicates().copy().groupby('v_enum').agg( { "age": "count", "gender": "count"} )
            df_counts.reset_index(inplace=True)
            df_counts = df_counts[
                (df_counts['age'].astype(int) == 2) | 
                (df_counts['gender'].astype(int) == 2)
            ]      # reduce that to only processible ones
            v_processable: "list[int]" = df_counts['v_enum'].unique().tolist()

            # now, work only on problem voices & records. For each voice, get related records and decide
            for v in v_processable:
                recs: pd.DataFrame = df[ df["v_enum"] == v ].copy()
                recs_blanks: pd.DataFrame = recs[ (recs["gender"] == const.NODATA) | (recs["age"] == const.NODATA) ].copy()          # get full blanks
                # gender
                recs_w_gender: pd.DataFrame = recs[ ~(recs["gender"] == const.NODATA) ].copy()
                if recs_w_gender.shape[0] > 0:
                    val: str = recs_w_gender["gender"].tolist()[0]
                    recs_blanks.loc[:, "gender"] = val
                # age
                recs_w_age: pd.DataFrame = recs[ ~(recs["age"] == const.NODATA) ].copy()
                if recs_w_age.shape[0] > 0:
                    val: str = recs_w_age["age"].tolist()[0]
                    recs_blanks.loc[:, "age"] = val
                # now we can add them to the result fixed list
                fixes = pd.concat( [ fixes.loc[:] , recs_blanks] ).reset_index(drop=True)

            # Here, we have a df maybe with records of possible changes
            if fixes.shape[0] > 0:
                # records
                pt: pd.DataFrame = pd.pivot_table(
                    fixes, values='p_enum', index=['age'], columns=['gender'], aggfunc='count',
                    fill_value=0, dropna=False, margins=False
                    )
                # get only value parts : nodata is just negative sum of these, and TOTAL will be 0,
                # so we drop them for file size and leave computation to the client
                pt = pt.reindex(const.CV_AGES, axis=0).reindex(const.CV_GENDERS, axis=1).fillna(
                    value=0).astype(int).drop(const.NODATA, axis=0).drop(const.NODATA, axis=1)
                dem_fixes_recs = arr2str(pt.to_numpy(int).tolist())

                # voices
                fixes = fixes.drop("p_enum", axis=1).drop_duplicates()
                pt: pd.DataFrame = pd.pivot_table(
                    fixes, values='v_enum', index=['age'], columns=['gender'], aggfunc='count',
                    fill_value=0, dropna=False, margins=False)
                # get only value parts : nodata is just -sum of these, sum will be 0
                pt = pt.reindex(const.CV_AGES, axis=0).reindex(const.CV_GENDERS, axis=1).fillna(
                    value=0).astype(int).drop(const.NODATA, axis=0).drop(const.NODATA, axis=1)
                dem_fixes_voices = arr2str(pt.to_numpy(int).tolist())

            return [dem_fixes_recs, dem_fixes_voices]

        # END - find_fixes

        #
        # === START ===
        #
        if DEBUG:
            print("ver=", ver, "lc=", lc, "algorithm=", algorithm, "split=", split)

        # Read in DataFrames
        if split != 'clips':
            df_orig: pd.DataFrame = df_read(fpath)
        else: # build "clips" from val+inval+other
            df_orig: pd.DataFrame = df_read(fpath) # we passed validated here, first read it.
            df2: pd.DataFrame = df_read(fpath.replace('validated', 'invalidated'))  # add invalidated
            df_orig = pd.concat( [df_orig.loc[:], df2] )
            df2: pd.DataFrame = df_read(fpath.replace('validated', 'other'))  # add other
            df_orig = pd.concat([df_orig.loc[:], df2])

        # Do nothing, if there is no data
        if df_orig.shape[0] == 0:
            results = {
                'ver':  ver,
                'lc':   lc,
                'alg':  algorithm,
                'sp':   split,
            }
            return results

        # Replace NA with NODATA
        df: pd.DataFrame = df_orig.fillna(value=const.NODATA)
        # add lowercase sentence column
        df['sentence_lower'] = df['sentence'].str.lower()

        #
        # DURATIONS
        #

        # Calc duration agregate values
        if df_clip_durations.shape[0] > 0:
            # Connect with duration table
            df["duration"] = df["path"].map(df_clip_durations["duration"])
            ser: pd.Series = df["duration"].dropna()
            duration_total: float = ser.sum()
            duration_mean: float = ser.mean()
            duration_median: float = ser.median()
            duration_std: float = ser.std(ddof=0)
            # Calc duration distribution
            arr: np.ndarray = np.fromiter(df["duration"].dropna().apply(
                int).reset_index(drop=True).to_list(), int)
            hist: list[list[int]] = np.histogram(arr, bins=const.BINS_DURATION)
            duration_freq: "list[int]" = hist[0]
            # duration_bins: "list[int]" = hist[1]
        else:  # No Duration data, set illegal defaults and continue
            duration_total: float = -1
            duration_mean: float = -1
            duration_median: float = -1
            duration_std: float = -1
            duration_freq: "list[int]" = []
            # duration_bins: "list[int]" = []

        #
        # VOICES
        #
        voice_counts: pd.DataFrame = df["client_id"].value_counts(
        ).dropna().to_frame().reset_index()
        voice_counts.rename(
            columns={"index": "voice", "client_id": "recordings"}, inplace=True)
        ser = voice_counts["recordings"]
        voice_mean: float = ser.mean()
        voice_median: float = ser.median()
        voice_std: float = ser.std(ddof=0)
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(voice_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_VOICES)
        voice_freq: "list[int]" = hist[0]
        # voice_bins: "list[int]" = hist[1]

        #
        # SENTENCES
        #
        sentence_counts: pd.DataFrame = df["sentence"].value_counts(
        ).dropna().to_frame().reset_index()
        sentence_counts.rename(
            columns={"index": "sentence", "sentence": "recordings"}, inplace=True)
        ser = sentence_counts["recordings"]
        sentence_mean: float = ser.mean()
        sentence_median: float = ser.median()
        sentence_std: float = ser.std(ddof=0)
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(sentence_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_SENTENCES)
        sentence_freq: "list[int]" = hist[0]
        # sentence_bins: "list[int]" = hist[1]

        #
        # VOTES
        #
        votes_counts: pd.DataFrame = df["up_votes"].value_counts(
        ).dropna().to_frame().reset_index()
        votes_counts.rename(
            columns={"index": "votes", "up_votes": "recordings"}, inplace=True)
        votes_counts = votes_counts.astype(int).reset_index(drop=True)
        ser = votes_counts["votes"]
        up_votes_mean: float = ser.mean()
        up_votes_median: float = ser.median()
        up_votes_std: float = ser.std(ddof=0)
        # FIXME This is bad coding
        up_votes_freq: "list[int]" = [0] * (len(const.BINS_VOTES_UP) -1)
        for i in range(0, len(const.BINS_VOTES_UP)-1):
            bin_val: int = const.BINS_VOTES_UP[i]
            bin_next: int = const.BINS_VOTES_UP[i+1]
            for inx, rec in votes_counts.iterrows():
                votes: int = rec["votes"]
                if ((votes >= bin_val) and (votes < bin_next)):
                    up_votes_freq[i] += rec["recordings"]


        votes_counts: pd.DataFrame = df["down_votes"].value_counts(
        ).dropna().to_frame().reset_index()
        votes_counts.rename(
            columns={"index": "votes", "down_votes": "recordings"}, inplace=True)
        ser = votes_counts["votes"]
        down_votes_mean: float = ser.mean()
        down_votes_median: float = ser.median()
        down_votes_std: float = ser.std(ddof=0)
        # FIXME This is bad coding
        down_votes_freq: "list[int]" = [0] * (len(const.BINS_VOTES_DOWN) -1)
        for i in range(0, len(const.BINS_VOTES_DOWN)-1):
            bin_val: int = const.BINS_VOTES_DOWN[i]
            bin_next: int = const.BINS_VOTES_DOWN[i+1]
            for inx, rec in votes_counts.iterrows():
                votes: int = rec["votes"]
                if ((votes >= bin_val) and (votes < bin_next)):
                    down_votes_freq[i] += rec["recordings"]

        #
        # BASIC MEASURES
        #
        clips_cnt: int = df.shape[0]
        unique_voices: int = df['client_id'].unique().shape[0]
        unique_sentences: int = df['sentence'].unique().shape[0]
        unique_sentences_lower: int = df['sentence_lower'].unique().shape[0]
        # Implement the following in the client: 
        # duplicate_sentence_cnt: int = clips_cnt - unique_sentences
        # duplicate_sentence_cnt_lower: int = clips_cnt - unique_sentences_lower


        #
        # DEMOGRAPHICS
        #

        # Add TOTAL to lists
        pt_ages: list[str] = const.CV_AGES.copy()
        pt_ages.append("TOTAL")
        pt_genders: list[str] = const.CV_GENDERS.copy()
        pt_genders.append("TOTAL")

        #
        # get a pt for all demographics (based on recordings)
        #
        _pt_dem: pd.DataFrame = pd.pivot_table(
            df, values='path', index=['age'], columns=['gender'], aggfunc='count',
            fill_value=0, dropna=False, margins=True, margins_name='TOTAL'
        )
        _pt_dem = _pt_dem.reindex(pt_ages, axis=0).reindex(pt_genders, axis=1).fillna(value=0).astype(int)

        #
        # get a pt for all demographics (based on unique voices)
        #
        _df_uqdem: pd.DataFrame = df[["client_id", "age", "gender"]]
        _df_uqdem = _df_uqdem.drop_duplicates().reset_index(drop=True)
        _pt_uqdem: pd.DataFrame = pd.pivot_table(
            _df_uqdem, values='client_id', index=['age'], columns=['gender'], aggfunc='count',
            fill_value=0, dropna=False, margins=True, margins_name='TOTAL'
        )
        _pt_uqdem = _pt_uqdem.reindex(pt_ages, axis=0).reindex(pt_genders, axis=1).fillna(value=0).astype(int)

        #
        # Create a table for all demographic info corrections (based on recordings)
        # Correctable ones are: clients with both blank and a single gender (or age) specified
        #
        dem_fixes_list: "list[str]" = find_fixes(df_orig)

        results: dict[str, Any] = {
            'ver':          ver,
            'lc':           lc,
            'alg':          algorithm,
            'sp':           split,

            'clips':        clips_cnt,
            'uq_v':         unique_voices,
            'uq_s':         unique_sentences,
            'uq_sl':        unique_sentences_lower,

            # Duration
            'dur_total':    round(1000 * duration_total) / 1000,
            'dur_avg':      round(1000 * duration_mean) / 1000,
            'dur_med':      round(1000 * duration_median) / 1000,
            'dur_std':      round(1000 * duration_std) / 1000,
            'dur_freq':     list2str(duration_freq),

            # Recordings per Voice
            'v_avg':        round(1000 * voice_mean) / 1000,
            'v_med':        round(1000 * voice_median) / 1000,
            'v_std':        round(1000 * voice_std) / 1000,
            'v_freq':       list2str(voice_freq),

            # Recordings per Sentence
            's_avg':        round(1000 * sentence_mean) / 1000,
            's_med':        round(1000 * sentence_median) / 1000,
            's_std':        round(1000 * sentence_std) / 1000,
            's_freq':       list2str(sentence_freq),

            # Votes
            'uv_avg':       round(1000 * up_votes_mean) / 1000,
            'uv_med':       round(1000 * up_votes_median) / 1000,
            'uv_std':       round(1000 * up_votes_std) / 1000,
            'uv_freq':      list2str(up_votes_freq),

            'dv_avg':       round(1000 * down_votes_mean) / 1000,
            'dv_med':       round(1000 * down_votes_median) / 1000,
            'dv_std':       round(1000 * down_votes_std) / 1000,
            'dv_freq':      list2str(down_votes_freq),

            # Demographics distribution for recordings
            'dem_table':    arr2str(_pt_dem.to_numpy(int).tolist()),
            'dem_uq':       arr2str(_pt_uqdem.to_numpy(int).tolist()),
            'dem_fix_r':    dem_fixes_list[0],
            'dem_fix_v':    dem_fixes_list[1],
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
        if os.path.isdir(lc_path): # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    lc_to_process: "list[str]" = DEBUG_CV_LC if DEBUG else lc_list

    cnt_datasets += len(lc_list)

    # Loop all locales
    cnt: int = 0
    res_all: "list[dict[str,Any]]" = []
    for lc in lc_to_process:
        cnt += 1
        res: "list[dict[str,Any]]" = []
        print(f'Processing {ver} locale {cnt}/{len(lc_to_process)} : {lc}')

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
            df_clip_durations: pd.DataFrame = df_read(
                cd_file).set_index("clip")
        else:
            print(f'WARNING: No duration data for {lc}\n')
            df_clip_durations: pd.DataFrame = pd.DataFrame(
                columns=const.COLS_CLIP_DURATIONS).set_index("clip")

        # MAIN SPLITS

        # Special case for temporary "clips.tsv"
        res.append(handle_split(
            ver=ver,
            lc=lc,
            algorithm="",
            split="clips",
            fpath=os.path.join(vc_dir, 'validated.tsv') # to start with we set validated
        ))

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
        res_all.extend(res)

    return res_all


########################################################
# Reported Stats
########################################################


# def reported_stats(lc_list: "list[str]") -> None:
#     pass

########################################################
# Text-Corpus Stats (Multi Processing Handler)
########################################################

def handle_text_corpus(lc: str) -> "dict[str,Any]":

    print(f'Processing text-corpus for locale: {lc}')

    tc_dir: str = os.path.join(HERE, "data", "text-corpus", lc)
 
    tc_file: str = os.path.join(tc_dir, '$text_corpus.tsv')
    tokens_file: str = os.path.join(tc_dir, '$tokens.tsv')

    # TEXT CORPUS

    # "file", "sentence", "lower", "normalized", "chars", "words", 'valid'
    df: pd.DataFrame = df_read(tc_file)

    # basic measures
    sentence_cnt: int = df.shape[0]
    unique_sentences: int = df['sentence'].dropna().unique().shape[0]
    unique_normalized: int = df['normalized'].dropna().unique().shape[0]
    valid: int = df[df['valid'].dropna().astype(int) == 1].shape[0]

    chars_total: int = df['chars'].dropna().sum()
    chars_mean: float = df['chars'].dropna().mean()
    chars_median: int = int(df['chars'].dropna().median())

    has_val: int = 0
    words_total: int = df['words'].dropna().sum()
    words_mean: float = 0
    words_median: int = 0
    if words_total != 0:
        has_val = 1
        words_mean: float = df['words'].dropna().mean()
        words_median: int = int(df['words'].dropna().median())

    # freq distributions

    # Calc character length distribution
    arr: np.ndarray = np.fromiter(df["chars"].dropna().apply(
        int).reset_index(drop=True).to_list(), int)
    hist: list[list[int]] = np.histogram(arr, bins=const.BINS_CHARS)
    character_freq: "list[int]" = hist[0]
    character_bins: "list[int]" = hist[1]

    # Calc word count distribution
    word_freq: "list[int]" = []
    if has_val == 1:
        arr: np.ndarray = np.fromiter(df["words"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_WORDS)
        word_freq: "list[int]" = hist[0]
        word_bins: "list[int]" = hist[1]

    # TOKENS

    tokens_total: int = 0
    tokens_mean: float = 0
    tokens_median: int = 0
    token_freq: "list[int]" = []
    if has_val == 1:
        df: pd.DataFrame = df_read(tokens_file)
        # "token", "count"
        tokens_total: int = df.shape[0]
        tokens_mean: float = df['count'].mean()
        tokens_median: int = int(df['count'].median())
        # Token/word repeat distribution
        arr: np.ndarray = np.fromiter(df["count"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_TOKENS)
        token_freq: "list[int]" = hist[0]
        token_bins: "list[int]" = hist[1]


    results: "dict[str, Any]" = {
        'lc': lc,
        's_cnt': sentence_cnt,
        'uq_s': unique_sentences,
        'uq_n': unique_normalized,
        'has_val': has_val,
        'val': valid,
        'c_total': chars_total,
        'c_mean': chars_mean,
        'c_median': chars_median,
        'c_freq': list2str(character_freq),
        'w_total': words_total,
        'w_mean': words_mean,
        'w_median': words_median,
        'w_freq': list2str(word_freq),
        't_total': tokens_total,
        't_mean': tokens_mean,
        't_median': tokens_median,
        't_freq': list2str(token_freq),
    }

    return results

########################################################
# Split Stats (Multi Processing Handler)
########################################################

def handle_version(cv_ver: str) -> "list[dict[str,Any]]":
    # Fins idx
    cv_idx: int = const.CV_VERSIONS.index(cv_ver)
    # Start Process
    return split_stats(cv_idx)

########################################################
# MAIN PROCESS
########################################################

def main() -> None:

    def ver2vercol(ver: str) -> str:
        """Converts a data version in format '11.0' to column/variable name format 'v11_0' """
        return 'v' + ver.replace('.', '_')


    used_proc_count: int = DEBUG_PROC_COUNT if DEBUG else PROC_COUNT
    print(f'=== Statistics Compilation Process for cv-tbox-dataset-analyzer ({used_proc_count} processes)===')
    start_time: datetime = datetime.now()


    #
    # MultiProcessing on text-corpora: Loop all locales, each locale in one process
    #
    print('\n=== Start Text Corpora Analysis ===\n')
    
    # First get list of languages with text corpora
    tc_base: str = os.path.join(HERE, "data", "text-corpus")
    lc_paths: "list[str]" = glob.glob(
        os.path.join(tc_base, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        if os.path.isdir(lc_path): # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    lc_to_process: "list[str]" = DEBUG_CV_LC if DEBUG else lc_list

    # Now multi-process each lc
    with mp.Pool(used_proc_count) as pool:
        tc_stats: list[dict[str, Any]] = pool.map(
            handle_text_corpus, lc_to_process)

    # Create result DF
    # df: pd.DataFrame = pd.DataFrame(tc_stats, columns=const.COLS_TEXT_CORPUS)
    df: pd.DataFrame = pd.DataFrame(tc_stats)
    df_write(df, os.path.join(
        HERE, "data", "results", 'tsv', '$text_corpus_stats.tsv'))
    df.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$text_corpus_stats.json'),
        orient='table', index=False)

    #
    # MultiProcessing on versions to handle splits: Loop all versions/languages/splits, each version in one process (TODO not ideal, should be refactored)
    #
    print('\n=== Start Dataset/Split Analysis ===\n')

    vers_to_process: "list[str]" = DEBUG_CV_VER if DEBUG else const.CV_VERSIONS

    with mp.Pool(used_proc_count) as pool:
        results: list[list[dict[str, Any]]] = pool.map(handle_version, vers_to_process)

    # done, first flatten it

    all_splits: list[dict[str, Any]] = []
    for res in results:
        all_splits.extend(res)

    # next form the support matrix
    df: pd.DataFrame = pd.DataFrame(all_splits).reset_index(drop=True)
    df = df[['ver', 'lc', 'alg']]
    df.drop_duplicates(['ver', 'lc', 'alg'], inplace=True)
    df.sort_values(['lc', 'ver', 'alg'], inplace=True)
    df = df[ df['alg'] != ""].reset_index(drop=True)

    # Prepare Support Matrix DataFrame
    rev_versions: "list[str]" = const.CV_VERSIONS.copy()
    rev_versions.reverse()
    for inx, ver in enumerate(rev_versions):
        rev_versions[inx] = ver2vercol(ver)
    COLS_SUPPORT_MATRIX: "list[str]" = ['lc', 'lang']
    COLS_SUPPORT_MATRIX.extend(rev_versions)
    df_support_matrix: pd.DataFrame = pd.DataFrame(
        index=const.ALL_LOCALES,
        columns=COLS_SUPPORT_MATRIX,
    )
    df_support_matrix['lc'] = const.ALL_LOCALES

    # Now loop and put the results inside
    for lc in const.ALL_LOCALES:
        for ver in const.CV_VERSIONS:
            subset: pd.DataFrame = df[(df['lc'] == lc ) & (df['ver'] == ver)]
            algo: "list[str]" = subset['alg'].unique().tolist()
            algos: str = const.SEP_ALGO.join(algo)
            df_support_matrix.at[lc, ver2vercol(ver)] = algos

    # Write out
    df_write(df_support_matrix, os.path.join(
        HERE, 'data', 'results', 'tsv', '$support_matrix.tsv'))
    df_support_matrix.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$support_matrix.json'),
        orient='table', index=False)

    # MORE TODO
    # Save config
    # Votes are problematic !!!
    # Fix DEM correction problem !!!
    # Get CV-Wide Datasets => Measures / Totals
    # Get var, std, mad values for some measures
    # Get global min/max/mean/median values for health measures
    # Get some statistical plots as images (e.g. corrolation: age-char speed graph)
    # 

    #
    # Save config
    #
    config_data: "dict[str, Any]" = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "cv_versions": const.CV_VERSIONS,
        "cv_dates": const.CV_DATES,
        "cv_locales": const.ALL_LOCALES,
        "algorithms": const.ALGORITHMS,
        "bins_duration": const.BINS_DURATION[:-1],          # Drop the last huge values from these lists
        "bins_voices": const.BINS_VOICES[:-1],
        "bins_votes_up": const.BINS_VOTES_UP[:-1],
        "bins_votes_down": const.BINS_VOTES_DOWN[:-1],
        "bins_sentences": const.BINS_SENTENCES[:-1],
        "bins_chars": const.BINS_CHARS[:-1],
        "bins_words": const.BINS_WORDS[:-1],
        "bins_tokens": const.BINS_TOKENS[:-1],
    }
    df: pd.DataFrame = pd.DataFrame([config_data]).reset_index(drop=True)
    # Write out
    df_write(df, os.path.join(
        HERE, 'data', 'results', 'tsv', '$config.tsv'))
    df.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$config.json'),
        orient='table', index=False)


    #
    # FINALIZE
    #
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'Finished compiling statistics for {df.shape[0]} datasets in {str(process_timedelta)} secs, avg={process_seconds/df.shape[0]} secs.')


if __name__ == '__main__':
    main()
